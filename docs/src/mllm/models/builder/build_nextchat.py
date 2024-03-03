import json
from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn

from ..nextchat.nextchat_seg import NextChatForSegLM
from ..nextchat.nextchat_base import NextChatForCausalLM


PREPROCESSOR = Dict[str, Any]

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_pretrained_nextchat_base(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]:
    model = NextChatForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        _fast_init=False,
        **kwargs
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.model.initialize_vision_modules(
        mm_depth=model_args.get("mm_projector_depth", 1),
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrained_mm_projector=model_args.pretrained_mm_projector,
        fsdp=training_args.fsdp,
    )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # HACK for quantization
    if model.model.get_vision_tower().device != torch.device('meta'):
        model.model.get_vision_tower().to(dtype=dtype, device=training_args.device)
    else:
        from transformers import CLIPVisionModel
        model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
        # model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip、
    vision_config = model_vision_dict['vision_config']


    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)


    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print(k)
    # TODO peft lora_model
    import json
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open("param.json", "w"))
    return model, preprocessor

def load_pretrained_nextchat(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]:
    model = NextChatForSegLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        _fast_init=False,
        sam_path=model_args.sam_path,
        # mm_vision_tower=model_args.vision_tower,
        **kwargs
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # model_vision_dict = model.model.initialize_vision_modules(
    #     vision_tower=model_args.vision_tower,
    #     mm_vision_select_layer=model_args.mm_vision_select_layer,
    #     fsdp=training_args.fsdp,
    # )
    model_vision_dict = model.model.initialize_vision_modules(
        mm_depth=model_args.get("mm_projector_depth", 1),
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrained_mm_projector=None,
        fsdp=training_args.fsdp,
    )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # HACK for quantization
    if model.model.get_vision_tower().device != torch.device('meta'):
        model.model.get_vision_tower().to(dtype=dtype, device=training_args.device)
    else:
        from transformers import CLIPVisionModel
        model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
        # model.model.vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip、
    vision_config = model_vision_dict['vision_config']


    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    # grad check
    model.requires_grad_(False)
    # model.model.vision_tower.requires_grad_(False)
    model.seg_prompt_mlp.requires_grad_(True)
    # model.sam.model.prompt_encoder.requires_grad_(True)
    # model.sam.requires_grad_(False)
    model.sam.model.mask_decoder.requires_grad_(True)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    # TODO peft lora_model
    import json
    json.dump({k: bool(v.requires_grad) for k, v in model.named_parameters()}, open("param.json", "w"))
    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
