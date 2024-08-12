import autorootcwd
import torch.nn as nn
import torch
from src.utils.registry import LOSS_REGISTRY
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss, SoftclDiceLoss

@LOSS_REGISTRY.register()
class MonaiDiceCELoss(nn.Module):
    """
    Wrapper for MONAI's DiceCELoss.
    
    This loss combines Dice loss and Cross-Entropy loss. It's useful for 
    segmentation tasks, balancing between region-based and pixel-wise loss.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = DiceCELoss(**kwargs)

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

@LOSS_REGISTRY.register()
class MonaiDiceFocalLoss(nn.Module):
    """
    Wrapper for MONAI's DiceFocalLoss.
    
    Combines Dice loss with Focal loss, which helps focus on hard-to-classify 
    examples. Good for datasets with class imbalance.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = DiceFocalLoss(**kwargs)

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

@LOSS_REGISTRY.register()
class MonaiSoftclDiceLoss(nn.Module):
    """
    Wrapper for MONAI's ClDiceLoss.
    
    Connectivity-preserving loss that combines Dice loss with a connectivity term.
    Useful for segmentation tasks where topological correctness is important.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = SoftclDiceLoss(**kwargs)

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)