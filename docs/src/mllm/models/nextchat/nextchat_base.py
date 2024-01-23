import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from mllm.utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor, \
    CLIPVisionConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_BOXES_TOKEN = "<boxes>"
DEFAULT_AT_TOKEN = "<at>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class NextChatConfig(LlamaConfig):
    model_type = "nextchat"
    sam_path = None
    mm_depth = 2


class NextChatLlamaModel(LlamaModel):
    config_class = NextChatConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(NextChatLlamaModel, self).__init__(config)

        # if hasattr(config, "mm_vision_tower"):
        #     vcfg = CLIPVisionConfig.from_pretrained(config.mm_vision_tower)
        #     self.vision_tower = CLIPVisionModel(vcfg)

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = self._build_mm_projector(config.mm_depth,
                                                         config.mm_hidden_size, config.hidden_size) # nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer, mm_depth=1,
                                  pretrained_mm_projector=None, fsdp=None):
        self.config.mm_vision_tower = vision_tower
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16) # TODO remove
        self.vision_tower = [vision_tower] if fsdp is not None and len(fsdp)>0 else vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = self._build_mm_projector(mm_depth,
                                                         vision_config.hidden_size, self.config.hidden_size)
            self.config.mm_depth=mm_depth
            # self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
        if pretrained_mm_projector is not None:
            logging.info(f"loading mm_projector from {pretrained_mm_projector}")
            mm_projector_weights = torch.load(pretrained_mm_projector, map_location='cpu')
            info = self.mm_projector.load_state_dict({k.replace("model.mm_projector.", ""): v for k, v in mm_projector_weights.items()})
            logging.info(info)
        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def get_vision_tower(self):
        vision_tower = self.vision_tower[0] if type(self.vision_tower) is list else self.vision_tower
        return vision_tower

    def _build_mm_projector(self, depth, in_size, out_size):
        if depth is None or depth<=1:
            return nn.Linear(in_size, out_size)

        modules = [nn.Linear(in_size, out_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_size, out_size))
        return nn.Sequential(*modules)

    def encode_input_embeds(self, input_ids, images, loc_embeds, orig_embeds_params, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = self.get_vision_tower()  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                            cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(),
                                                              cur_input_embeds[image_start_token_pos:image_start_token_pos + 1],
                                                              cur_image_features, cur_input_embeds[
                                                                                  image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                                              cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos + 1], cur_image_features,
                                                              cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches, device=masked_indices.device,
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start + num_patches:]),
                            dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # add the loc embeddings into the input_embeds
        if (input_ids == vision_tower.config.box_token).sum() > 0 and loc_embeds is not None:
            inputs_embeds[input_ids == vision_tower.config.box_token] = loc_embeds.type(inputs_embeds.dtype)
        return inputs_embeds


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            loc_embeds: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.encode_input_embeds(input_ids, images, loc_embeds, orig_embeds_params, inputs_embeds)

        return super(NextChatLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class NextChatForCausalLM(LlamaForCausalLM):
    config_class = NextChatConfig

    def __init__(self, config: NextChatConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = NextChatLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loc_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        self.loc_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 4)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            loc_inputs=None,
            loc_targets=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        vision_tower = self.model.get_vision_tower()
        if labels is not None:
            labels[labels==vision_tower.config.box_token] = -100

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # TODO change to loc_inputs
        loc_embeds = None
        if loc_inputs is not None and len(loc_inputs) > 0:
            loc_embeds = self.loc_encoder(loc_inputs)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            images=images,
            loc_embeds=loc_embeds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:-1][..., :-1, :].contiguous()
            shift_labels = labels[:-1][..., 1:].contiguous()
            # x, y = shift_labels[shift_labels > 29871], shift_logits.argmax(-1)[shift_labels > 29871]
            # print((x==y).sum()/len(x))
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)


        pred_locs = None
        cycle_loss1= None
        if loc_targets is not None and len(loc_targets)>0:
            last_hidden_states = outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))
            loc_positions = ( (input_ids.flatten() == vision_tower.config.at_token)
                             & (labels.flatten()>0) ).nonzero().flatten()
            selected_hidden_states = last_hidden_states[loc_positions]
            # pred_locs = self.loc_mlp(selected_hidden_states)
            pred_locs = self.loc_decoder(selected_hidden_states)
            # pred_locs = F.relu(pred_locs)
            # loc_targets_cxcywh = box_xyxy_to_cxcywh(loc_targets)
            if len(pred_locs) != len(loc_targets):
                torch.save([input_ids, labels, attention_mask, loc_inputs, loc_targets], "tmp.pth")
            box_loss = self.box_loss(pred_locs, loc_targets)
            loss += box_loss
            print(torch.diag(box_iou(pred_locs, loc_targets)[0]).mean())

            # cycle loss
            pred_output_embeds = self.loc_encoder(pred_locs)
            cycle_loss1 = F.mse_loss(pred_output_embeds, selected_hidden_states, reduction="none")
            cycle_loss1 = self.masked_loss(cycle_loss1, 1)
            loss += cycle_loss1
            # print()

        # cycle loss
        if loc_embeds is not None:
            pred_input_locs = self.loc_decoder(loc_embeds)
            cycle_loss2 = F.l1_loss(pred_input_locs, loc_inputs, reduction="none")
            cycle_loss2 = self.masked_loss(cycle_loss2, 1)
            loss += cycle_loss2

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def box_loss(self, src_boxes, target_boxes):
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = self.masked_loss(loss_bbox, 1)

        mask = (src_boxes[:, 2:] >= src_boxes[:, :2]).all(-1)
        src_boxes = src_boxes[mask]
        target_boxes = target_boxes[mask]
        # if not mask.all():
        #     print(len(mask)-mask.sum())

        loss_giou = 1 - torch.diag(generalized_box_iou(
            src_boxes,
            target_boxes))
        loss_giou = self.masked_loss(loss_giou, 1)
        return loss_bbox*2 + loss_giou/5

    def masked_loss(self, loss, n):
        mask = torch.ones_like(loss)
        mask[-n:] = 1e-10
        loss = (loss*mask).sum()/(mask.sum())
        return loss


    def generate_rec(
        self,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        use_cache = kwargs["use_cache"]
        images = kwargs["images"]

        to_append = torch.tensor([673, 29901, 32003, 32004,   29889,     2], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, to_append.repeat(len(input_ids), 1)], 1)
        to_append_attn = torch.ones([len(attention_mask), len(to_append)], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, to_append_attn], 1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            images=images.type(self.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype)
        )
        last_hidden_states = outputs.hidden_states[-1]
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))
        vision_tower = self.model.get_vision_tower()
        loc_positions = (input_ids.flatten() == vision_tower.config.at_token).nonzero().flatten()
        selected_hidden_states = last_hidden_states[loc_positions]
        pred_locs = self.loc_decoder(selected_hidden_states)
        pred_locs = pred_locs
        return pred_locs


    # def prepare_inputs_for_generation(
    #         self, input_ids, loc_inputs=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     if past_key_values:
    #         loc_ids = None
    #         if input_ids.size(-1)>=2:
    #             loc_ids = input_ids[:, -2]
    #         input_ids = input_ids[:, -1:]
    #
    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         inputs_embeds = self.model.embed_tokens(input_ids)
    #         hidden_states = kwargs.pop("hidden_states", None)
    #         vision_tower = self.model.get_vision_tower()
    #         # need to incorporate location information
    #         if loc_ids is not None and (loc_ids==vision_tower.config.at_token).any():
    #             mask = loc_ids==vision_tower.config.at_token
    #             loc_embeds = hidden_states[-1][mask][:, -1:, :]
    #             loc_embeds = loc_embeds.type(inputs_embeds.dtype)
    #             pred_locs = self.loc_decoder(loc_embeds)
    #             loc_embeds = self.loc_encoder(pred_locs)
    #             inputs_embeds[mask] = loc_embeds
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #
    #     model_inputs.update(
    #         {
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #             "images": kwargs.get("images", None),
    #             "loc_inputs": loc_inputs,
    #         }
    #     )
    #     return model_inputs

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        standardize_cache_format=False,
    ):
        model_kwargs = super(NextChatForCausalLM, self)._update_model_kwargs_for_generation(outputs,
                                                                                model_kwargs,
                                                                                is_encoder_decoder,
                                                                                standardize_cache_format)
        model_kwargs.update({"hidden_states": outputs.hidden_states})
        return model_kwargs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_tower = self.model.get_vision_tower()
        vision_config = vision_tower.config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_AT_TOKEN, DEFAULT_BOXES_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token, vision_config.at_token, vision_config.box_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_AT_TOKEN, DEFAULT_BOXES_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 3
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
