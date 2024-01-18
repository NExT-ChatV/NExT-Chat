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

# losses from LISA
def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


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

        self.bce_loss_weight = 2
        self.dice_loss_weight = 0.5
    # def forward(self, pred_masks, gt_masks, iou_predictions, device):
    #     loss_focal = 0.
    #     loss_dice = 0.
    #     loss_iou = 0.
    #     num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
    #     for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
    #         gt_mask = gt_mask.to(device)
    #         batch_iou = calc_iou(pred_mask, gt_mask)
    #         loss_focal += self.focal_loss(pred_mask, gt_mask, num_masks)
    #         loss_dice += self.dice_loss(pred_mask, gt_mask, num_masks)
    #         loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
    #
    #     loss_total = 20. * loss_focal + loss_dice + loss_iou
    #     return loss_total
    def forward(self, pred_masks, gt_masks, iou_predictions, device):
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
            # Compute Binary Cross-Entropy Loss
            mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                              gt_mask.shape[0])
            # Compute Dice Loss
            mask_dice_loss += (
                    calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
            num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        return mask_loss
