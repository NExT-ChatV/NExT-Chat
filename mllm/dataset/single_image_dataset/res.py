import random
import sys
import logging
import warnings
from typing import Dict, Any, Sequence
from pycocotools import mask as maskutil
import numpy as np

import torch
from torchvision.ops import box_iou

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def decode_mask(mask, h, w):
    rle = maskutil.frPyObjects(mask, h, w)
    m = maskutil.decode(rle)
    m = np.sum(m, axis=2)
    m = m.astype(np.uint8)  # convert to np.uint8
    return m


@DATASETS.register_module()
class RESDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = random.choice(item['expressions'])
        h, w = item["height"], item['width']
        bbox = item['bbox']
        mask = decode_mask(item['mask'], h, w)

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'masks': [mask],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {BOXES_PLACEHOLDER} .',
                    'boxes_seq': [[0]],
                }
            ]
        }
        return ret


@METRICS.register_module()
class RESComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def __call__(self, eval_preds) -> Dict[str, Any]:
        preds, targets = eval_preds
        ciou = preds.sum()/targets.sum()
        fails = (preds<=0).sum()
        return {
            'ciou': ciou,
            'failed': fails,
        }
        # logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        # assert len(preds) == len(targets)
        # return self.calculate_metric(preds, targets)
    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes = preds
        target_boxes = targets
        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
