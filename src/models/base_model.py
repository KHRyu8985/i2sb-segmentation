import autorootcwd
import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import src.metrics.brats21 as metrics
from src.metrics.brats21 import brats_post_processing
from tqdm import tqdm

import abc

class BaseModel():
    """Base model class for training segmentation models """
    
    def __init__(self, arch, device, aux_arch=None, mode='train'):
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"
        
        self.arch = arch
        self.device = device
        self.mode = mode
        
        self.arch.to(self.device)
        
        if aux_arch:
            # this is for additional networks such as scorer from meta-weight-net or etc.
            self.aux_arch = aux_arch
            self.aux_arch.to(self.device)

    def make_dataloaders(self, train_loader=None, val_loader=None, test_loader=None):
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        if test_loader is not None:
            self.test_loader = test_loader

        if self.mode == 'train':
            if not hasattr(self, 'train_loader') or not hasattr(self, 'val_loader'):
                raise ValueError("For 'train' mode, both train_loader and val_loader must be provided.")
    
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
            'dice_wt': 0.0,
            'dice_tc': 0.0,
            'dice_et': 0.0,
            'hd95': 0.0,
            'hd95_wt': 0.0,
            'hd95_tc': 0.0,
            'hd95_et': 0.0
        }
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=f"Validation Epoch {current_epoch}/{total_epoch}") as pbar:
                for batch_idx, batch in enumerate(dataloader):
                    img, label = self.feed_data(batch)
                    pred_seg = self.arch(img)
                    loss = self.criterion(pred_seg, label)
                    
                    running_loss += loss.item()
                    
                    pred_seg = pred_seg > 0.5  # Convert to boolean after inference
                    pred_seg = brats_post_processing(pred_seg)                    
                    batch_metrics = self.compute_metrics(pred_seg, label)
                            
                    for key in running_metrics:
                        running_metrics[key] += batch_metrics[key]
                    
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
                    
        epoch_loss = running_loss / len(dataloader)
        avg_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}
        
        print(
            f"Validation Epoch [{current_epoch}/{total_epoch}], Loss: {epoch_loss:.4f}, "
            f"Dice: {avg_metrics['dice']:.4f}, Dice_WT: {avg_metrics['dice_wt']:.4f}, "
            f"Dice_TC: {avg_metrics['dice_tc']:.4f}, Dice_ET: {avg_metrics['dice_et']:.4f}, "
            f"HD95: {avg_metrics['hd95']:.4f}, HD95_WT: {avg_metrics['hd95_wt']:.4f}, "
            f"HD95_TC: {avg_metrics['hd95_tc']:.4f}, HD95_ET: {avg_metrics['hd95_et']:.4f}"
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
        dice = metrics.dice(seg_map, label.bool())
        hd95 = metrics.hd95(seg_map, label.bool())
        
        # case by case
        dice_wt = dice[:,1]
        dice_tc = dice[:,0]
        dice_et = dice[:,2]
        
        hd95_wt = hd95[:,1]
        hd95_tc = hd95[:,0]
        hd95_et = hd95[:,2]
        
        dice = dice.mean()
        dice_wt = dice_wt.mean()
        dice_tc = dice_tc.mean()
        dice_et = dice_et.mean()
        
        hd95 = hd95.mean()
        hd95_wt = hd95_wt.mean()
        hd95_tc = hd95_tc.mean()
        hd95_et = hd95_et.mean()
        
        return {
            'dice': dice,
            'dice_wt': dice_wt,
            'dice_tc': dice_tc,
            'dice_et': dice_et,
            'hd95': hd95,
            'hd95_wt': hd95_wt,
            'hd95_tc': hd95_tc,
            'hd95_et': hd95_et
        }


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