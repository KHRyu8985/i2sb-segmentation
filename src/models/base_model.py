import autorootcwd
import os
import src.data
import src.archs
from src.utils.registry import ARCH_REGISTRY

import time
import torch
import src.metrics.vessel_2d as metrics
from tqdm import tqdm

import abc

class BaseModel():
    """Base model class for training segmentation models """
    
    def __init__(self, arch='FRNet', device='cuda:0', mode='train'):
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"
        assert arch in ARCH_REGISTRY, f"Architecture {arch} not found in ARCH_REGISTRY! Available architectures: {ARCH_REGISTRY.keys()}"
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"

        self.arch = ARCH_REGISTRY.get(arch)(in_channels=1, out_channels=1) # Could be FRNet, AttentionUNet, SegResNet
        self.mode = mode # train, infer
        self.device = device
    
    @abc.abstractmethod
    def feed_data(self, batch):
        # how is input and label data derived from batch
        ''' Example
        img, label = batch['image'], batch['label']
        img = img.to(self.device)
        label = label.to(self.device)
        '''
        pass
    
    @abc.abstractmethod
    def train_step(self, batch):
        """Train step.
        Args:
            batch : batch
        """ 
        pass
    
    @abc.abstractmethod
    def train_one_epoch(self, dataloader, current_epoch, total_epoch=0):
        pass

    def val_one_epoch(self, dataloader, current_epoch, total_epoch=0):
        self.arch.eval()
        running_loss = 0.0
        running_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'hausdorff_distance': 0.0,
            'clDice': 0.0
        }
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=f"Validation Epoch {current_epoch}/{total_epoch}") as pbar:
                for batch_idx, batch in enumerate(dataloader):
                    img, label = self.feed_data(batch)
                    pred_seg = self.arch(img)
                    loss = self.criterion(pred_seg, label)
                    
                    running_loss += loss.item()
                    
                    pred_seg = pred_seg > 0.5  # Convert to boolean after inference
                    batch_metrics = self.compute_metrics(pred_seg, label)
                            
                    for key in running_metrics:
                        running_metrics[key] += batch_metrics[key]
                    
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
                    
        epoch_loss = running_loss / len(dataloader)
        avg_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}
        
        print(
            f"Validation Epoch [{current_epoch}/{total_epoch}], Loss: {epoch_loss:.4f}, "
            f"Dice: {avg_metrics['dice']:.4f}, IoU: {avg_metrics['iou']:.4f}, "
            f"Accuracy: {avg_metrics['accuracy']:.4f}, Sensitivity: {avg_metrics['sensitivity']:.4f}, "
            f"Specificity: {avg_metrics['specificity']:.4f}, Hausdorff Distance: {avg_metrics['hausdorff_distance']:.4f}, "
            f"clDice: {avg_metrics['clDice']:.4f}"
        )
        

    def train_val_one_epoch(self, current_epoch=1, total_epoch=1):
        start_time = time.time()
        if total_epoch ==1 and current_epoch == 1:
            print("Starting training and validation for one epoch.")
    
        self.train_one_epoch(self.train_loader, current_epoch, total_epoch)
        self.val_one_epoch(self.val_loader, current_epoch, total_epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        if total_epoch ==1 and current_epoch == 1:
            print(f"Completed training and validation for one epoch. Time taken: {elapsed_time:.2f} seconds")

    
    def get_current_visuals(self):
        pass
    
    def get_current_log(self):
        pass
    
    def save(self, epoch):
        """Save networks and training state."""
        pass
    
    def compute_metrics(self, seg_map, label):
        metrics_dict = metrics.calculate_all_metrics(seg_map, label)
        return metrics_dict


    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def prepare_training(self, optimizer, train_dataloader, lr_scheduler=None):
        self.arch, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.arch, optimizer, train_dataloader, lr_scheduler
        )
        self.train_loader = train_dataloader
        return optimizer, lr_scheduler

    def print_model_summary(self):
        print("Model Summary:")
        total_params = 0
        for name, param in self.arch.named_parameters():
            total_params += param.numel()
            print(f"Layer: {name}, Size: {param.size()}")
        print(f"Total Parameters: {total_params}")
        
        if hasattr(self, 'aux_arch'):
            # Add the code you want to run if aux_arch is defined
            print("Auxiliary Model Summary:")
            aux_total_params = 0
            for name, param in self.aux_arch.named_parameters():
                aux_total_params += param.numel()
                print(f"Aux Layer: {name}, Size: {param.size()}")
            print(f"Auxiliary Total Parameters: {aux_total_params}")