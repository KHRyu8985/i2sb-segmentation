from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch.nn.functional as F

def multi_class_score(one_class_fn, predictions, labels, num_classes, one_hot=False):
    result = np.zeros(num_classes)

    for label_index in range(num_classes):
        if one_hot:
            class_predictions = predictions[:, label_index, ...]
            class_labels = labels[:, label_index, ...]
        else:
            class_predictions = predictions.eq(label_index)         # prediction [batch, 1, x,y,x]  dim2 = 0-num_class   --> class_prediction= either0 or 1
            class_predictions = class_predictions.squeeze(1)  # remove channel dim
            class_labels = labels.eq(label_index)
            class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()
        class_labels = class_labels.float()

        result[label_index] = one_class_fn(class_predictions, class_labels).mean()

    return result


def hausdorff_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    def one_class_hausdorff_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_hausdorff_distance, predictions, labels, num_classes=num_classes)


def average_surface_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    def one_class_average_surface_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetAverageHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_average_surface_distance, predictions, labels, num_classes=num_classes)


def dice_score(predictions, labels, num_classes, one_hot=False):
    """ returns the dice score

    Args:
        predictions: one hot tensor [B, num_classes, D, H, W]
        labels: label tensor [B, 1, D, H, W]
    Returns:
        dict: ['label'] = [B, score]
    """

    def one_class_dice(pred, lab,smooth=1e-5):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()

        return (2. * true_positive+smooth) / (p_flat.sum() + l_flat.sum()+smooth)

    return multi_class_score(one_class_dice, predictions, labels, num_classes=num_classes, one_hot=one_hot)

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def dice_loss(pred, lab,smooth=1e-5):
    activation = nn.Sigmoid()
    pred = activation(pred)
    shape = pred.shape
    p_flat = pred.view(shape[0], -1)
    l_flat = lab.view(shape[0], -1)
    true_positive = (p_flat * l_flat).sum()

    return 1- (2. * true_positive+ smooth) / (p_flat.sum() + l_flat.sum()+ smooth)

def precision(predictions, labels, num_classes):
    def one_class_precision(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        return true_positive / p_flat.sum()

    return multi_class_score(one_class_precision, predictions, labels, num_classes=num_classes)

def specificity(predictions, labels, num_classes):
    def one_class_specificity(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_neagtive = ((1 - p_flat) * (1- l_flat)).sum()
        false_positive = (p_flat * (1- l_flat)).sum() # labeled as negative but predict as positive
        return true_neagtive / (true_neagtive + false_positive)

    return multi_class_score(one_class_specificity, predictions, labels, num_classes=num_classes)


def recall(predictions, labels, num_classes):
    def one_class_recall(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        negative = 1 - p_flat
        false_negative = (negative * l_flat).sum()
        return true_positive / (true_positive + false_negative)

    return multi_class_score(one_class_recall, predictions, labels, num_classes=num_classes)

def dice_metric(predictions, targets, smooth=1e-5):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_metric(predictions, targets, smooth=1e-5):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def accuracy_metric(predictions, targets):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    correct = (predictions == targets).float().sum()
    total = torch.numel(predictions)
    return correct / total

def sensitivity_metric(predictions, targets, smooth=1e-5):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    true_positives = (predictions * targets).sum()
    actual_positives = targets.sum()
    return (true_positives + smooth) / (actual_positives + smooth)

def specificity_metric(predictions, targets, smooth=1e-5):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    true_negatives = ((1 - predictions) * (1 - targets)).sum()
    actual_negatives = (1 - targets).sum()
    return (true_negatives + smooth) / (actual_negatives + smooth)

def cl_dice_metric(predictions, targets):
    def cl_score(v, s):
        return np.sum(v*s)/np.sum(s)

    if isinstance(predictions, torch.Tensor):
        v_p = predictions.squeeze().cpu().numpy()
    else:
        v_p = predictions.squeeze()
    
    if isinstance(targets, torch.Tensor):
        v_l = targets.squeeze().cpu().numpy()
    else:
        v_l = targets.squeeze()
    
    v_p = v_p.astype(bool)
    v_l = v_l.astype(bool)

    if v_p.ndim == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif v_p.ndim == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    else:
        raise ValueError("Unsupported dimension for clDice calculation")

    return 2 * tprec * tsens / (tprec + tsens)

def calculate_all_metrics(predictions, targets, spacing=[1, 1, 1]):
    # Convert to float if necessary
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(np.float32)
    else:
        predictions = predictions.float()
    
    if isinstance(targets, np.ndarray):
        targets = targets.astype(np.float32)
    else:
        targets = targets.float()
    
    # For binary segmentation, we need to threshold the predictions
    if isinstance(predictions, np.ndarray):
        predictions = (predictions > 0.5).astype(np.float32)
    else:
        predictions = (predictions > 0.5).float()
    
    # Calculate metrics
    dice = dice_metric(predictions, targets)
    iou = iou_metric(predictions, targets)
    accuracy = accuracy_metric(predictions, targets)
    sensitivity = sensitivity_metric(predictions, targets)
    specificity = specificity_metric(predictions, targets)
    cl_dice = cl_dice_metric(predictions, targets)
    
    return {
        "Dice": dice.item() if isinstance(dice, torch.Tensor) else dice,
        "IoU": iou.item() if isinstance(iou, torch.Tensor) else iou,
        "Accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
        "Sensitivity": sensitivity.item() if isinstance(sensitivity, torch.Tensor) else sensitivity,
        "Specificity": specificity.item() if isinstance(specificity, torch.Tensor) else specificity,
        "clDice": cl_dice
    }