import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # inputs = F.sigmoid(inputs)
        # inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    # batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


class SamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred_masks, gt_masks, iou_predictions, device):
        loss_focal = 0.
        loss_dice = 0.
        loss_iou = 0.
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
            gt_mask = gt_mask.to(device)
            batch_iou = calc_iou(pred_mask, gt_mask)
            loss_focal += self.focal_loss(pred_mask, gt_mask, num_masks)
            loss_dice += self.dice_loss(pred_mask, gt_mask, num_masks)
            loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

        loss_total = 20. * loss_focal + loss_dice + loss_iou
        return loss_total