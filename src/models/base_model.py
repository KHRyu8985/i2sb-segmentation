import autorootcwd
import os
import src.data
import src.archs
from src.utils.registry import ARCH_REGISTRY, LOSS_REGISTRY

import time
import torch
import src.metrics.vessel_2d as metrics
from tqdm import tqdm

import abc

class BaseModel():
    """Base model class for training segmentation models """
    
    def __init__(self, arch='FRNet', criterion='DiceCELoss',device='cuda:0', mode='train'):
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"
        assert arch in ARCH_REGISTRY, f"Architecture {arch} not found in ARCH_REGISTRY! Available architectures: {ARCH_REGISTRY.keys()}"
        assert mode in ['train', 'infer'], f"mode should be either 'train' or 'infer', but got {mode}"

        self.arch = ARCH_REGISTRY.get(arch)(in_channels=1, out_channels=1) # Could be FRNet, AttentionUNet, SegResNet
        self.criterion = LOSS_REGISTRY.get(criterion)() # Could be DiceCELoss, DiceLoss, CrossEntropyLoss
        self.mode = mode # train, infer
        self.device = device
        self.first_verbose = True  # Add this flag to track the first verbose plotting
    
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
    def valid_step(self, batch):
        """Validation step.
        Args:
            batch : batch
        """ 
        pass

    def train_one_epoch(self, dataloader, current_epoch, total_epoch=0):
        self.arch.train()
        running_loss = 0.0
        weighted_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch}/{total_epoch}")
        for i, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            running_loss += loss
            weighted_loss = 0.8 * weighted_loss + 0.2 * loss
            progress_bar.set_postfix(loss=f"{weighted_loss:.3f}")
        return running_loss / len(dataloader)

    def val_one_epoch(self, dataloader, current_epoch, total_epoch=0, verbose=True, base_folder='results/unit_test'):
        self.arch.eval()
        running_loss = 0.0
        running_metrics = {
            'Dice': 0.0,
            'IoU': 0.0,
            'Accuracy': 0.0,
            'Sensitivity': 0.0,
            'Specificity': 0.0,
            'clDice': 0.0
        }
        best_dice = [-1, -1]
        worst_dice = [float('inf'), float('inf')]
        best_indices = [-1, -1]
        worst_indices = [-1, -1]
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=f"Validation Epoch {current_epoch}/{total_epoch}") as pbar:
                for batch_idx, batch in enumerate(dataloader):
                    
                    pred_seg, label = self.valid_step(batch)
                    loss = self.criterion(pred_seg, label)
                    
                    running_loss += loss.item()
                    
                    pred_seg = pred_seg > 0.5  # Convert to boolean after inference
                    batch_metrics = self.compute_metrics(pred_seg, label)
                            
                    for key in running_metrics:
                        running_metrics[key] += batch_metrics[key]
                    
                    dice_score = batch_metrics['Dice']
                    if dice_score > best_dice[0]:
                        best_dice[1] = best_dice[0]
                        best_indices[1] = best_indices[0]
                        best_dice[0] = dice_score
                        best_indices[0] = batch_idx
                    elif dice_score > best_dice[1]:
                        best_dice[1] = dice_score
                        best_indices[1] = batch_idx

                    if dice_score < worst_dice[0]:
                        worst_dice[1] = worst_dice[0]
                        worst_indices[1] = worst_indices[0]
                        worst_dice[0] = dice_score
                        worst_indices[0] = batch_idx
                    elif dice_score < worst_dice[1]:
                        worst_dice[1] = dice_score
                        worst_indices[1] = batch_idx
                    
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
                    
        epoch_loss = running_loss / len(dataloader)
        avg_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}
        
        print(
            f"Validation Epoch [{current_epoch}/{total_epoch}], Loss: {epoch_loss:.4f}, "
            f"Dice: {avg_metrics['Dice']:.4f}, IoU: {avg_metrics['IoU']:.4f}, "
            f"Accuracy: {avg_metrics['Accuracy']:.4f}, Sensitivity: {avg_metrics['Sensitivity']:.4f}, "
            f"Specificity: {avg_metrics['Specificity']:.4f}, clDice: {avg_metrics['clDice']:.4f}"
        )
        print(f"Best Indices: {best_indices}, Worst Indices: {worst_indices}")
        
        if verbose:
            import os
            import matplotlib.pyplot as plt

            # Ensure the base folder exists
            os.makedirs(base_folder, exist_ok=True)

            fig, axes = plt.subplots(2, 6, figsize=(24, 10))  # Adjusted the figure size for better width and height
            best_batches = []
            worst_batches = []
            for idx, batch in enumerate(dataloader):
                if idx in best_indices:
                    best_batches.append(batch)
                if idx in worst_indices:
                    worst_batches.append(batch)
            # Now you can use best_batches and worst_batches as needed
            
            for i in range(2):
                best_batch = best_batches[i]
                worst_batch = worst_batches[i]
                
                best_img, best_label = self.feed_data(best_batch)
                best_pred = self.arch(best_img)
                best_pred = torch.sigmoid(best_pred) > 0.5
                
                worst_img, worst_label = self.feed_data(worst_batch)
                worst_pred = self.arch(worst_img)
                worst_pred = torch.sigmoid(worst_pred) > 0.5

                axes[0, i*3].imshow(best_img.cpu().squeeze(), cmap='gray')
                axes[0, i*3].axis('off')
                axes[0, i*3].set_title(f'Best Image {i+1} (Index: {best_indices[i]}) Dice: {best_dice[i]:.3f}')
                axes[0, i*3+1].imshow(best_pred.cpu().squeeze(), cmap='gray')
                axes[0, i*3+1].axis('off')
                axes[0, i*3+1].set_title('Best Prediction')
                axes[0, i*3+2].imshow(best_label.cpu().squeeze(), cmap='gray')
                axes[0, i*3+2].axis('off')
                axes[0, i*3+2].set_title('Best Ground Truth')

                axes[1, i*3].imshow(worst_img.cpu().squeeze(), cmap='gray')
                axes[1, i*3].axis('off')
                axes[1, i*3].set_title(f'Worst Image {i+1} (Index: {worst_indices[i]}) Dice: {worst_dice[i]:.3f}')
                axes[1, i*3+1].imshow(worst_pred.cpu().squeeze(), cmap='gray')
                axes[1, i*3+1].axis('off')
                axes[1, i*3+1].set_title('Worst Prediction')
                axes[1, i*3+2].imshow(worst_label.cpu().squeeze(), cmap='gray')
                axes[1, i*3+2].axis('off')
                axes[1, i*3+2].set_title('Worst Ground Truth')

            plt.tight_layout()
            plt.savefig(os.path.join(base_folder, f'validation_epoch_{current_epoch}.png'))
            plt.close()
            
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