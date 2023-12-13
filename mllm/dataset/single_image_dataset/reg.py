import json

from ..utils import (
    MInstrDataset,
)

from ..root import (
    METRICS,
    DATASETS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
)

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from pycocoevalcap.eval import Cider, Meteor, PTBTokenizer
import sys
import warnings
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

@DATASETS.register_module()
class REGDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression'] if "expression" in item else item["expressions"][0]
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
        caption = expr

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GCDataset(REGDataset):
    pass


@METRICS.register_module()
class REGCapComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def calculate_metric(self, preds, targets):
        preds = [self.extract_ans(p) for p in preds]
        preds = {i: [{"caption": x}] for i, x in enumerate(preds)}

        targets = [self.extract_ans(t) for t in targets]
        targets = {i: [{"caption": x}] for i, x in enumerate(targets)}
        json.dump({"preds": preds, "targets": targets}, open("rst.json", "w"))

        tokenizer = PTBTokenizer()
        targets  = tokenizer.tokenize(targets)
        preds = tokenizer.tokenize(preds)
        json.dump({"preds": preds, "targets": targets}, open("rst.json", "w"))
        cider_score, meteor_score = Cider(), Meteor()
        cider_rst, _ = cider_score.compute_score(targets, preds)
        meteor_rst, _ = meteor_score.compute_score(targets, preds)

        return {
            "CIDEr": cider_rst*100,
            "Meteor": meteor_rst,
        }

    def extract_ans(self, string: str):
        try:
            string = string.split("ASSISTANT: ")[-1].lower().split("</s>")[0]
            return string
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None
