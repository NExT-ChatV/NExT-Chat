import json
import numbers
import os
import re
import sys
import logging
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Any, Union

import torch
import numpy as np
import gradio as gr
from PIL import Image
from PIL import ImageDraw, ImageFont
from mmengine import Config
import transformers
# from transformers import BitsAndBytesConfig
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.utils import draw_bounding_boxes, ImageBoxState, bbox_draw, open_image, parse_boxes
from mllm.models.builder.build_nextchat import load_pretrained_nextchat

log_level = logging.ERROR
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)

#########################################
# mllm model init
#########################################
def build_model(model_name_or_path, vit_model_path, image_token_len=256, load_in_8bit=False):
    model_args = Config(dict(
        type='nextchat',
        version='v1',

        # checkpoint config
        cache_dir=None,
        model_name_or_path=model_name_or_path,
        vision_tower=vit_model_path,
        pretrain_mm_mlp_adapter=None,
        sam_path=None,

        # model config
        mm_vision_select_layer=-2,
        model_max_length=2048,

        # finetune config
        freeze_backbone=False,
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,

        # data process config
        is_multimodal=True,
        sep_image_conv_front=False,
        image_token_len=image_token_len,
        mm_use_im_start_end=True,

        target_processor=dict(
            boxes=dict(type='PlainBoxFormatter'),
        ),

        process_func_args=dict(
            conv=dict(type='ChatConvProcess'),
            target=dict(type='BoxFormatProcess'),
            text=dict(type='ChatTextProcess'),
            image=dict(type='ChatImageProcessor'),
        ),

        conv_args=dict(
            conv_template='vicuna_v1.1',
            transforms=dict(type='Expand2square'),
            tokenize_kwargs=dict(truncation_size=None),
        ),

        gen_kwargs_set_pad_token_id=True,
        gen_kwargs_set_bos_token_id=True,
        gen_kwargs_set_eos_token_id=True,
    ))
    training_args = Config(dict(
        bf16=True,
        fp16=False,
        device='cuda',
        fsdp=None,
    ))

    # if load_in_8bit:
    #     quantization_kwargs = dict(
    #         quantization_config=BitsAndBytesConfig(
    #             load_in_8bit=True,
    #         )
    #     )
    # else:
    #     quantization_kwargs = dict()
    quantization_kwargs = dict()

    model, preprocessor = load_pretrained_nextchat(model_args, training_args, **quantization_kwargs)
    if not getattr(model, 'is_quantized', False):
        model.to(dtype=torch.bfloat16, device=torch.device('cuda'))
    if not getattr(model.model.get_vision_tower(), 'is_quantized', False):
        model.model.get_vision_tower().to(dtype=torch.bfloat16, device=torch.device('cuda'))
    print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    print(f"vision device: {model.model.get_vision_tower().device}, is_quantized: {getattr(model.model.get_vision_tower(), 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

    preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    tokenizer = preprocessor['text']
    return model, model_args, preprocessor, tokenizer


#########################################
# demo utils
#########################################

def parse_text(text):
    text = text.replace("<image>", "&lt;image&gt;")
    return text


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


def resize_pil_img(pil_img: Image.Image, *, w, h):
    old_height, old_width = pil_img.height, pil_img.width
    new_height, new_width = (h, w)
    if (new_height, new_width) == (old_height, old_width):
        return pil_img
    return pil_img.resize((new_width, new_height))


def resize_box_xyxy(boxes, *, w, h, ow, oh):
    old_height, old_width = (oh, ow)
    new_height, new_width = (h, w)
    if (new_height, new_width) == (old_height, old_width):
        return boxes
    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    out_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = x1 * w_ratio
        x2 = x2 * w_ratio
        y1 = y1 * h_ratio
        y2 = y2 * h_ratio
        nb = (x1, y1, x2, y2)
        out_boxes.append(nb)
    return out_boxes


def binarize(x):
    return (x != 0).astype('uint8') * 255


def de_transform_mask(orgw, orgh, mask):
    long_side = max(orgw, orgh)
    short_side = min(orgw, orgh)
    pad = (long_side - short_side) // 2
    mask = F.interpolate(mask, [long_side, long_side], mode="bilinear", align_corners=False)
    mask = mask > 0
    mask[mask > 0] = 255
    if orgw < orgh:
        mask = mask[..., :, pad: short_side + pad]
    else:
        mask = mask[..., pad: short_side + pad, :]
    # mask = mask.transpose(2, 3)
    # print(mask.shape)
    return mask.squeeze(1)


def de_occlude_masks(masks):
    union_mask = torch.zeros_like(masks[0]).int()
    for i, m in enumerate(masks):
        m[m.int() + union_mask.int() >= 2] = 0
        print((m.int() + union_mask.int() >= 2).sum())
        union_mask = m.int() + union_mask
    return masks


def de_transform_box(orgw, orgh, boxes):
    long_side = max(orgw, orgh)
    short_side = min(orgw, orgh)
    pad = (long_side - short_side) // 2
    boxes = boxes * long_side
    if orgw < orgh:
        boxes[:, 0] -= pad
        boxes[:, 2] -= pad
    else:
        boxes[:, 1] -= pad
        boxes[:, 3] -= pad
    return boxes


def draw_boxes(img, _boxes, texts, colors):
    assert img is not None
    _img_draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'assets/DejaVuSansMono.ttf'), size=36)
    for bid, box in enumerate(_boxes):
        _img_draw.rectangle((box[0], box[1], box[2], box[3]), outline=colors[bid % len(colors)], width=8)
        anno_text = texts[bid]
        _img_draw.rectangle(
            (box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]),
            outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=8)
        _img_draw.text((box[0] + int(font.size * 0.2), box[3] - int(font.size * 1.2)), anno_text, font=font,
                       fill=(255, 255, 255))
    return img


def post_process_response(response):
    if "<at> <boxes>" not in response:
        return response.replace("<", "&lt;").replace(">", "&gt;")
    splits = response.split("<at> <boxes>")
    to_concat = [f"[{i}]" for i in range(len(splits) - 1)]
    rst = [splits[i // 2] if i % 2 == 0 else to_concat[i // 2]
           for i in range(len(splits) + len(to_concat))]
    rst = "".join(rst)
    rst = rst.replace("<", "&lt;").replace(">", "&gt;")
    return rst


def model_predict(model, model_args, tokenizer, preprocessor, image, text,
                  temperature=0.75, top_p=0.7, top_k=5, boxes=None, boxes_seq=None, iou_thres=0.3):
    image = open_image(image)
    orgw, orgh = image.width, image.height
    conversation = prepare_interactive(model_args, preprocessor)
    conversation.set_image(image)
    conversation.append_message(role=conversation.roles[0],
                                message=text, boxes=boxes, boxes_seq=boxes_seq)
    inputs = conversation.to_model_input()

    inputs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
    output_ids, masks, iou_predictions, boxes = model.generate(**inputs)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # print(response)
    # print(boxes)

    filename_grounding = None
    ret_image = None
    if boxes is not None:
        # remove low quality masks
        # print(masks.shape, iou_predictions.squeeze().shape)
        for idx, iou in enumerate(iou_predictions.squeeze(1)):
            if iou < iou_thres:
                masks[idx, :] = 0

        boxes = de_transform_box(orgw, orgh, boxes)
        masks = de_transform_mask(orgw, orgh, masks)
        # masks = de_occlude_masks(masks)

        colors = ["#F76566", "#18ACBA", "#9400D3", "#454926", "#4E72B8"]
        if len(colors) < len(boxes):
            colors += ["#F76566"] * (len(boxes) - len(colors))

        from torchvision.transforms import PILToTensor, ToPILImage
        Timage = PILToTensor()(image)

        from torchvision.utils import draw_segmentation_masks
        res = draw_segmentation_masks(Timage, masks, colors=colors, alpha=0.5)

        res = ToPILImage()(res)
        res = draw_boxes(res, boxes, [str(i) for i in range(len(boxes))], colors)
        ret_image = res
        # timestamp = int(time.time())
        # filename_grounding = f"tmp/sever_imgs/{timestamp}.jpg"
        # if not os.path.exists("tmp/sever_imgs/"):
        #     os.makedirs("tmp/sever_imgs/")
        # res.save(filename_grounding)
    return response, boxes, masks, ret_image


class NextChatInference(object):
    def __init__(self, model_path, vit_path, image_token_len=576, **kwargs):
        self.model, self.model_args, self.preprocessor, self.tokenizer = build_model(model_path, vit_path, image_token_len=image_token_len)

    def __call__(self, input_tensor, **forward_params):
        image, text = input_tensor["image"], input_tensor["text"]
        temperature = forward_params.pop("temperature", 0.8)
        top_p = forward_params.pop("top_p", 0.7)
        top_k = forward_params.pop("top_k", 5)
        boxes = forward_params.pop("boxes", [])
        boxes_seq = forward_params.pop("boxes_seq", [])
        iou_thres = forward_params.pop("iou_thres", 0.3)

        # check for parsing box
        cur_input_text, cur_boxes_seq = parse_boxes(text)
        if len(cur_boxes_seq) > 0:
            text = cur_input_text
            boxes_seq = cur_boxes_seq
        self.model.eval()
        response, boxes, masks, ret_img = model_predict(self.model, self.model_args,
                                          self.tokenizer, self.preprocessor,
                                          image, text,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          iou_thres=iou_thres,
                                          boxes=boxes, boxes_seq=boxes_seq)
        return response, boxes, masks, ret_img