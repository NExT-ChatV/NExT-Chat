import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINER_STATE_NAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class TrainerDifferentCollatorMixin:
    def __init__(self,
                 *args,
                 train_collator: Optional[DataCollator] = None,
                 eval_collator: Optional[DataCollator] = None,
                 test_collator: Optional[DataCollator] = None,
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
            warnings.warn('[WARNING!!!] use different collator for eval and test. but maybe do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.) u should'
                          'check your code and know exactly what u are doing.')
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
        self._test_collator = test_collator if test_collator is not None else self._eval_collator
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_train_dataloader(self) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader


# noinspection DuplicatedCode
class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        # if self.is_local_process_zero():
        #     print(self.lr_scheduler.lr_lambdas[0])
        #     print(self.lr_scheduler.last_epoch,
        #           self.lr_scheduler.lr_lambdas[0](self.lr_scheduler.last_epoch),
        #           not self.accelerator.optimizer_step_was_skipped)
        return inputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom behavior.

        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs:
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys())
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                if len(inputs["loc_targets"])==len(inputs["loc_inputs"]) and len(inputs['loc_targets'])>0 and "masks_sam" not in inputs: # eval rec
                    generated_tokens = self.model.generate_rec(**gen_kwargs)
                    # generated_tokens = self.tensor2token(generated_tokens)
                else:
                    generated_tokens = self.model.generate(**gen_kwargs)
                    if "masks_sam" not in inputs and type(generated_tokens) is tuple:
                        generated_tokens = generated_tokens[0]
                    # generated_tokens = self.tensor2token(generated_tokens)

        # TODO: rewrite official seq2seq_trainer to suppress generation_config warning
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # important for Decoder-Only LLM: only extract generated_tokens and discard origin inputs
        generation_inputs = inputs['input_ids']
        # generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        # if generated_tokens.shape[-1] < gen_config.max_length:
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        # elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            if "masks_sam" in inputs: # res segmentation eval
                gt_masks = inputs["masks_sam"]
                gt_masks = torch.stack(gt_masks, 0)
                if generated_tokens[1] is None:
                    pred_masks = gt_masks.clone()
                    pred_masks[:] = 0
                    print("fail")
                else:
                    pred_masks = generated_tokens[1]
                pred_masks = self.de_transform_mask(inputs["img_size"][0, 1], inputs["img_size"][0, 0], pred_masks)
                gt_masks = inputs["unresized_masks"][0].unsqueeze(0)
                if (pred_masks.size()!=gt_masks.size()):
                    pred_masks = gt_masks.clone()
                    pred_masks[:] = 0
                    print("unmatched")
                intersection = torch.sum(torch.mul(pred_masks, gt_masks), dim=(1, 2))
                union = torch.sum(pred_masks, dim=(1, 2)) + torch.sum(gt_masks, dim=(1, 2)) - intersection
                generated_tokens = intersection
                labels = union

            elif len(inputs["loc_targets"]) > 0: # rec bounding box eval
                # labels = self.tensor2token(inputs["loc_targets"])
                labels = inputs["loc_targets"]
            else: # image2text gen task eval
                labels = inputs["labels"]
            # if labels.shape[-1] < gen_config.max_length:
            #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            # elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
            #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None
        assert len(generated_tokens) == len(labels)
        return loss, generated_tokens, labels

    def de_transform_mask(self, orgw, orgh, mask):
        long_side = max(orgw, orgh)
        short_side = min(orgw, orgh)
        pad = (long_side - short_side) // 2
        mask = F.interpolate(mask, [long_side, long_side], mode="bilinear", align_corners=False)
        mask = mask > 0
        mask[mask > 0] = 1
        mask[mask<=0] = 0
        if orgw < orgh:
            mask = mask[..., :, pad: short_side + pad]
        else:
            mask = mask[..., pad: short_side + pad, :]
        # mask = mask.transpose(2, 3)
        # print(mask.shape)
        return mask.squeeze(1)

    def tensor2token(self, tensor_list):
        lst = [str(tensor.cpu().tolist()) for tensor in tensor_list]
        tokens = self.tokenizer(lst, return_tensors="pt", add_special_tokens=False, padding="longest")
        return tokens.input_ids.to(tensor_list[0].device)

    def _logging_generate_kwargs(self, keys):
        if not hasattr(self, '_generate_kwargs'):
            self._generate_kwargs = None
        if self._generate_kwargs != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def save_prediction(self, predict_results, file_key_prefix='predict'):
        if not self.is_world_process_zero():
            return

        import numpy as np
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)

        preds, targets = predict_results.predictions, predict_results.label_ids
        origin_preds, origin_targets = preds, targets
        preds, targets = deepcopy(preds), deepcopy(targets)
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")

        # decode text and save to json takes forever for big test set
        os.makedirs(self.args.output_dir, exist_ok=True)
        # with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
        #     for p, t, pi, ti in tqdm(
        #             zip(preds, targets, origin_preds, origin_targets),
        #             total=len(preds), desc=f"saving prediction for {file_key_prefix}",
        #     ):
        #         p[p < 0] = self.tokenizer.pad_token_id
        #         t[t < 0] = self.tokenizer.pad_token_id
        #         p = self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         t = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         obj = dict(
        #             pred=p,
        #             target=t,
        #             # pred_id=pi.tolist(),
        #             # target_id=ti.tolist(),
        #         )
        #         g.write(json.dumps(obj) + '\n')
        #         g.flush()

    # transformers + FSDP + saving model -> cuda OOM for small memory gpu
    # refer: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.fsdp is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=cpu_state_dict)  # noqa
            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            super().save_model(output_dir, _internal_call)

    def plot_loss(self) -> None:
        if not self.is_world_process_zero():
            return

        training_args = self.args
        FIGURE_NAME = "trainer_state.png"
        import matplotlib.pyplot as plt
        data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
        train_steps, train_losses = [], []
        for i in range(len(data["log_history"]) - 1):
            train_steps.append(data["log_history"][i]["step"])
            train_losses.append(data["log_history"][i]["loss"])
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.title("training loss of {}".format(training_args.output_dir))
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
        print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


class Seq2SeqDataCollator(DataCollatorForSeq2Seq):
    def __init__(
            self,
            inference_mode: bool = False,
            **kwargs,
    ):
        self.inference_mode = inference_mode
        self.text_keys = ['input_ids', 'labels', 'attention_mask']
        super().__init__(**kwargs)

    def __call__(self, features: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
        # evaluation/inference adopts left-padding while training adopts right-padding
        text_features = [{k: feature[k] for k in self.text_keys if k in feature} for feature in features]

        if self.inference_mode:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side
        else:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'right'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side

        return text_features


class Seq2Seq2DataCollatorWithImage(Seq2SeqDataCollator):
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)
        # sometimes there is either no location input or output in the current batch
        # which will make some parameters untrained in the batch.
        # use a mock annotation to prevent error
        self.mock = torch.load("mock.pth")

    # noinspection PyMethodMayBeStatic
    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [feature['image'] for feature in features]
        images = torch.stack(images, dim=0)
        ret = dict(images=images)
        return ret

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if not self.inference_mode and not ("masks_sam" in features[0]):
            features.append(self.mock)
        loc_inputs = [x['loc_inputs'] for x in features]
        loc_targets = [x['loc_targets'] for x in features]
        ret = super().__call__(features, return_tensors)
        image_outputs = self._image_process(features)
        ret.update(image_outputs)
        ret["loc_inputs"] = torch.tensor([list(a) for b in loc_inputs for a in b])
        ret["loc_targets"] = torch.tensor([list(a) for b in loc_targets for a in b])

        if "images_sam" in features[0]:
            ret['images_sam'] = torch.stack([f["images_sam"] for f in features], dim=0)
        if "masks_sam" in features[0]:
            ret['masks_sam'] = [torch.stack(f["masks_sam"], 0) for f in features]
            ret['img_size'] = torch.stack([f["img_size"] for f in features], 0)
            ret['unresized_masks'] = [f["unresized_masks"] for f in features]
            # ret['masks_sam'] = [y for x in ret["masks_sam"] for y in x]
        return ret
