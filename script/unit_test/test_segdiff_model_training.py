import autorootcwd
import torch

import src.data
import src.losses
import src.metrics
import src.models

from src.models.segdiff_model import SegDiffModel
from src.utils.registry import DATASET_REGISTRY, ARCH_REGISTRY, LOSS_REGISTRY
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from src.metrics.vessel_2d import calculate_all_metrics
import numpy as np
import os
from torch.utils.data import Subset

# Test the dataset
#path_OCTA500_6M = "data/OCTA500_6M"
#path_OCTA500_3M = "data/OCTA500_3M"
path_ROSSA = "data/ROSSA"

# Test create_segmentation_dataset
path_dict = {
#    "OCTA500_6M": path_OCTA500_6M,
#    "OCTA500_3M": path_OCTA500_3M,
    "ROSSA": path_ROSSA
}

# Test with collapse=True
datasets_collapsed = DATASET_REGISTRY.get(
    'VesselDataset')(path_dict, collapse=True)
print("Collapsed datasets:")
for split, dataset in datasets_collapsed.items():
    print(f"{split}: {len(dataset)} samples")


train_dataset = datasets_collapsed['train']
valid_dataset = datasets_collapsed['val']

valid_indices = list(range(20))
valid_dataset = Subset(valid_dataset, valid_indices) # 20개만으로 validation dataset 줄임 (시간을 위해)

test_dataset = datasets_collapsed['test']

print('Length of datasets:')
print(len(train_dataset), len(valid_dataset), len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=20, shuffle=False,num_workers=0)

#printing the arch and criterion available in the registry
print('Available arch in the registry:')
print(ARCH_REGISTRY.keys())

print('Available loss in the registry:')
print(LOSS_REGISTRY.keys())

Model = SegDiffModel(
    arch='SegDiffUnet', criterion='MonaiDiceFocalLoss', mode='train') # check the available arch and criterion in the registry

Model.to('cuda:0')

# Perform train and val one epoch
for epoch in range(100):
    print(f"Epoch {epoch+1}")
    Model.train_one_epoch(train_dataloader, epoch+1, 100)
    Model.train_one_epoch(train_dataloader, epoch+1, 100)
    Model.train_one_epoch(train_dataloader, epoch+1, 100)
    Model.train_one_epoch(train_dataloader, epoch+1, 100)
    Model.train_one_epoch(train_dataloader, epoch+1, 100)
    Model.train_one_epoch(train_dataloader, epoch+1, 100)

    if epoch % 5 == 0:
        if epoch % 10 == 0: # skip the first 10 epochs
            Model.val_one_epoch(valid_dataloader, epoch+1, 100, verbose=True) # if verbose = True, it will plot the images
        else:
            Model.val_one_epoch(valid_dataloader, epoch+1, 100, verbose=False) # else, it will not plot the images
