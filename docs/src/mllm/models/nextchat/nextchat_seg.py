from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig

from mllm.models.nextchat.nextchat_base import NextChatForCausalLM, NextChatConfig
from mllm.models.sam.modeling_sam import SamForLMSeg
from mllm.models.sam.sam_loss import SamLoss


class NextChatForSegLM(NextChatForCausalLM):
    def __init__(self, config: NextChatConfig):
        super(NextChatForSegLM, self).__init__(config)
        self.sam = SamForLMSeg("vit_h", config.sam_path)
        self.sam_loss = SamLoss()
        self.sam_prompt_dim = self.sam.model.prompt_encoder.embed_dim
        self.seg_prompt_mlp = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.model.config.hidden_size, self.sam_prompt_dim*4)
            )

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
            images_sam: Optional[torch.FloatTensor] = None,
            masks_sam: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            loc_inputs=None,
            loc_targets=None, # mask
            **kwargs,
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

        loss = 0
        if loc_targets is not None and len(loc_targets) > 0:
            last_hidden_states = outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))
            loc_positions = ( (input_ids.flatten() == vision_tower.config.at_token)
                             & (labels.flatten()>0) ).nonzero().flatten()
            selected_hidden_states = last_hidden_states[loc_positions]
            pred_locs = self.loc_decoder(selected_hidden_states)*1024

            prompt_states = self.seg_prompt_mlp(selected_hidden_states)
            prompt_states = prompt_states.view(prompt_states.size(0), -1, self.sam_prompt_dim)
            pred_masks, iou_predictions = self.sam(images_sam, prompt_states, pred_locs)
            seg_loss = self.sam_loss(pred_masks, masks_sam, iou_predictions, last_hidden_states.device)
            loss += seg_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        inputs= None,
        generation_config= None,
        logits_processor= None,
        stopping_criteria= None,
        prefix_allowed_tokens_fn= None,
        synced_gpus= None,
        streamer= None,
        images_sam= None,
        **kwargs,
    ):
        dtype = self.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        images = kwargs["images"]
        loc_inputs = kwargs.pop("loc_inputs", "None")
        loc_embeds = None
        if loc_inputs is not None and len(loc_inputs)>0:
            loc_embeds = self.loc_encoder(loc_inputs.type(dtype))
            vision_tower = self.model.get_vision_tower()
            num = (input_ids==vision_tower.config.box_token).sum()
            loc_embeds = loc_embeds[:num]
            if num == 0:
                loc_embeds = None

        orig_embeds_params = getattr(self.model, 'orig_embeds_params', None)
        input_embeds = self.model.encode_input_embeds(input_ids, images.type(dtype), loc_embeds,
                                                      orig_embeds_params, inputs_embeds=None)

        outputs = super(NextChatForCausalLM, self).generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.pop("max_new_tokens", 1024),
            # stopping_criteria=stopping_criteria,
            num_beams=kwargs.pop("num_beams", 5),
            min_length=1,
            top_p=kwargs.get("top_p", 0.8),
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=kwargs.get("temperature", 0.75),
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            top_k=kwargs.get("top_k", 5),
        )

        loc_hidden_states = []
        if hasattr(outputs, "beam_indices"): # beam size > 1
            vision_tower = self.model.get_vision_tower()
            loc_ids = (outputs.sequences == vision_tower.config.at_token).nonzero()
            hidden_states = outputs.hidden_states
            beam_indices = outputs.beam_indices

            for lid in loc_ids:
                # assign to box
                outputs.sequences[lid[0], lid[1]+1] = vision_tower.config.box_token
                beam_idx = beam_indices[lid[0], lid[1]]
                loc_h = hidden_states[lid[1]][-1][beam_idx]
                loc_hidden_states.append(loc_h.squeeze())
            if len(loc_hidden_states) > 0:
                loc_hidden_states = torch.stack(loc_hidden_states)
        else: # beam_size == 1
            vision_tower = self.model.get_vision_tower()
            loc_ids = (outputs.sequences == vision_tower.config.at_token).nonzero()
            hidden_states = outputs.hidden_states
            for lid in loc_ids:
                outputs.sequences[lid[0], lid[1]+1] = vision_tower.config.box_token
                loc_h = hidden_states[lid[1]][-1]
                loc_hidden_states.append(loc_h.squeeze())
            if len(loc_hidden_states) > 0:
                loc_hidden_states = torch.stack(loc_hidden_states)

        pred_masks, pred_locs, iou_predictions = None, None, None
        if len(loc_hidden_states)>0:
            loc_hidden_states = loc_hidden_states.type(dtype)
            pred_locs = self.loc_decoder(loc_hidden_states)

            prompt_states = self.seg_prompt_mlp(loc_hidden_states)
            prompt_states = prompt_states.view(prompt_states.size(0), -1, self.sam_prompt_dim)
            dtype = self.sam.model.image_encoder.patch_embed.proj.weight.dtype
            if images_sam is not None:
                pred_masks, iou_predictions = self.sam(images_sam.type(dtype), prompt_states, boxes=pred_locs.type(dtype)*1024)
        return outputs.sequences, pred_masks, iou_predictions, pred_locs

    def prepare_inputs_for_generation(
            self, input_ids, images_sam=None, loc_inputs=None,
            past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            loc_ids = None
            if input_ids.size(-1)>=2:
                loc_ids = input_ids[:, -2]
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)
            hidden_states = kwargs.pop("hidden_states", None)
            vision_tower = self.model.get_vision_tower()
            # need to incorporate location information
            if loc_ids is not None and (loc_ids==vision_tower.config.at_token).any():
                mask = loc_ids==vision_tower.config.at_token
                loc_embeds = hidden_states[-1][mask][:, -1:, :]
                loc_embeds = loc_embeds.type(inputs_embeds.dtype)

                pred_locs = self.loc_decoder(loc_embeds)
                loc_embeds = self.loc_encoder(pred_locs)
                inputs_embeds[mask] = loc_embeds
            model_inputs = {"inputs_embeds": inputs_embeds}
            # model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "loc_inputs": loc_inputs,
                "images_sam": images_sam,
            }
        )
        return model_inputs