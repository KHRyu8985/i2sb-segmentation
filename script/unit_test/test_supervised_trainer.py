import autorootcwd
import src.data
import src.losses
import src.metrics
import src.models

import os
import csv

from src.models.supervised_model import SupervisedModel
from src.utils.registry import DATASET_REGISTRY, ARCH_REGISTRY, LOSS_REGISTRY
from torch.utils.data import DataLoader
from src.utils.trainer import Trainer, Inferer
from src.metrics.vessel_2d import calculate_all_metrics

import numpy as np
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
test_dataset = datasets_collapsed['test']

print('Length of datasets:')
print(len(train_dataset), len(valid_dataset), len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=8, shuffle=False)

#printing the arch and criterion available in the registry
print('Available arch in the registry:')
print(ARCH_REGISTRY.keys())

print('Available loss in the registry:')
print(LOSS_REGISTRY.keys())

model = SupervisedModel(arch='AttentionUnet', criterion='MonaiDiceFocalLoss', mode='train')
model_folder_name = model.get_name()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    train_num_steps=5000,
    valid_every=100,
    save_every=50
)
trainer.train()


"""
model_folder_name = model.get_name()

print('TESTING RESUME')
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    train_num_steps=5000,
    valid_every=100,
    save_every=500,
    resume_from=os.path.join('results',model_folder_name)
)
trainer.train()
"""