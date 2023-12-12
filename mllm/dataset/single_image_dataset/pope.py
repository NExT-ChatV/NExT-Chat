from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..root import DATASETS, IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER, QUESTION_PLACEHOLDER, METRICS
from ..utils import MInstrDataset, BaseComputeMetrics


@DATASETS.register_module()
class POPEVQADataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['image'])

        question = item['text']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        label = str(item['label']).lower()

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label} .",
                },
            ]
        }
        return ret

import re
ANS_EXTRACT_PAT = re.compile(r'(?:(?:(?:(?:(?:So t)|(?:T)|(?:t))he answer is)|(?:Answer:)) (.+))')

@METRICS.register_module()
class PopeComputeMetrics(BaseComputeMetrics):
    def extract_ans(self, string: str):
        if "ASSISTANT: " in string:
            string = string.split("ASSISTANT: ")[-1].lower()
        try:
            found = ANS_EXTRACT_PAT.findall(string.strip())
            if len(found) != 1:
                if "yes" in string:
                    return "yes"
                else:
                    return "no"
            return found[0].strip().rstrip('.').strip()
        except (IndexError, AttributeError):
            return None

    def calculate_metric(self, preds, targets):
        correct = 0
        failed = 0
        target_failed = 0
        preds = [self.extract_ans(pred) for pred in preds]
        targets = [self.extract_ans(target) for target in targets]
        acc, precision, recall, f1, yes_ratio = self.other_metric(preds, targets)
        for pred, target in zip(preds, targets):
            extract_pred = pred
            extract_target = target
            if extract_target is None:
                target_failed += 1
                continue
            if extract_pred is None:
                failed += 1
            if extract_pred == extract_target:
                correct += 1
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            "f1": f1,
            "yes_ratio": yes_ratio,
            'target_failed': target_failed,
            'failed': failed,
        }

    def other_metric(self, answers, label_list):
        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc, precision, recall, f1, yes_ratio
