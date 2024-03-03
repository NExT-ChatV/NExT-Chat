# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""

from typing import Dict, Any, Tuple, Optional

from PIL import Image

from ..root import TRANSFORMS

import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from mllm.utils.box_ops import box_xyxy_to_cxcywh
from mllm.utils.box_ops import interpolate


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def box_xywh_to_xyxy(box, *, w=None, h=None):
    x, y, bw, bh = box
    x2 = x + bw
    y2 = y + bh
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = x, y, x2, y2
    return box


def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    # normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    normalized_box = (norm_x1, norm_y1, norm_x2, norm_y2)
    return normalized_box


def norm_point_xyxy(point, *, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


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


def point_xy_expand2square(point, *, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point


@TRANSFORMS.register_module()
class Expand2square:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Tuple[Image.Image, Optional[Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if 'boxes' in labels:
            bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['boxes']]
            labels['boxes'] = bboxes
        if 'points' in labels:
            points = [point_xy_expand2square(point, w=width, h=height) for point in labels['points']]
            labels['points'] = points
        return processed_image, labels




def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    if target is None:
        return cropped_image, None, True

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = torch.tensor(target["boxes"])
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    all_box_reserved = True
    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        # for field in fields:
        #     target[field] = target[field][keep]
        all_box_reserved = keep.all()

    return cropped_image, target, all_box_reserved


def hflip(image, target):
    flipped_image = F.hflip(image)

    if target is None:
        return flipped_image, None

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = torch.tensor(target["boxes"])
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = torch.tensor(target["boxes"])
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target



class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        for i in range(3):
            rst_img, rst_target, all_box_reserved = crop(img, target, region)
            if all_box_reserved:
                return rst_img, rst_target
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

@TRANSFORMS.register_module()
class TrainAug:
    def __init__(self, background_color=(255, 255, 255)):
        self.resize_ratio = 0.25
        self.background_color = background_color
        self.random_size_crop = Compose(
                                            [
                                                RandomResize([400, 500, 600]),
                                                RandomSizeCrop(384, 1333),
                                            ]
                                        )
        self.random_horizontal = RandomHorizontalFlip()

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Tuple[Image.Image, Optional[Dict[str, Any]]]:
        # if random.random() > self.resize_ratio:
        #     image, labels = self.random_size_crop(image, labels)
        image, labels = self.random_horizontal(image, labels)

        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if 'boxes' in labels:
            bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['boxes']]
            labels['boxes'] = bboxes
        if 'points' in labels:
            points = [point_xy_expand2square(point, w=width, h=height) for point in labels['points']]
            labels['points'] = points
        return processed_image, labels