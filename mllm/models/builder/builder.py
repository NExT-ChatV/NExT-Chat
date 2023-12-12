from typing import Dict, Any, Tuple

from torch import nn
from .build_nextchat import load_pretrained_nextchat, load_pretrained_nextchat_base

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if type_ == 'nextchat':
        return load_pretrained_nextchat_base(model_args, training_args)
    elif type_ == "nextchat_seg":
        return load_pretrained_nextchat(model_args, training_args)
    else:
        assert False
