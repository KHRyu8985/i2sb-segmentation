import autorootcwd
import src.data
from src.utils.registry import DATASET_REGISTRY
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from src.metrics.vessel_2d import calculate_all_metrics
import numpy as np

# Test the dataset
path_OCTA500_6M = "data/OCTA500_6M"
path_OCTA500_3M = "data/OCTA500_3M"
path_ROSSA = "data/ROSSA"

# Test create_segmentation_dataset
path_dict = {
    "OCTA500_6M": path_OCTA500_6M,
    "OCTA500_3M": path_OCTA500_3M,
    "ROSSA": path_ROSSA
}

# Test with collapse=True
datasets_collapsed = DATASET_REGISTRY.get('VesselDataset')(path_dict, collapse=True)
print("Collapsed datasets:")
for split, dataset in datasets_collapsed.items():
    print(f"{split}: {len(dataset)} samples")

# Test with collapse=False
datasets_uncollapsed = DATASET_REGISTRY.get('VesselDataset')(path_dict, collapse=False)
print("\nUncollapsed datasets:")
print(f"Train: {len(datasets_uncollapsed['train'])} samples")
print(f"Val: {len(datasets_uncollapsed['val'])} samples")
print("Test:")
for name, dataset in datasets_uncollapsed['test'].items():
    print(f"  {name}: {len(dataset)} samples")

# Test a single item from the training set
train_dataset = datasets_collapsed['train']
image, label, name = train_dataset[0]
print(f"\nSample item - Name: {name}")
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")

# Create and test DataLoader for collapsed dataset
print("\nTesting DataLoader for dataset:")
train_dataloader_collapsed = DataLoader(datasets_collapsed['train'], batch_size=4, shuffle=True)
batch = next(iter(train_dataloader_collapsed))
images, labels, names = batch

# Introduce errors only in the positive labels (1s)
error_rate = 0.2
error_mask = (torch.rand_like(labels) < error_rate) & (labels == 1)
labels_with_error = labels.clone()
labels_with_error[error_mask] = 0

# Calculate metrics
print(labels.shape, labels_with_error.shape)
metrics = calculate_all_metrics(labels, labels_with_error)

# Print metrics
print("\nCalculated Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

# Plot images, labels, and labels with error
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
for i in range(4):
    axes[i, 0].imshow(images[i, 0].numpy(), cmap='gray')
    axes[i, 0].set_title(f"Image: {names[i]}")
    axes[i, 1].imshow(labels[i, 0].numpy(), cmap='gray')
    axes[i, 1].set_title("Original Label")
    axes[i, 2].imshow(labels_with_error[i, 0].numpy(), cmap='gray')
    axes[i, 2].set_title("Label with Error")

    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()
plt.show()