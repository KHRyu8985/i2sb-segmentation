import autorootcwd
import src.data
import src.archs
from src.utils.registry import DATASET_REGISTRY, ARCH_REGISTRY
import torch
import matplotlib.pyplot as plt
from src.metrics.vessel_2d import calculate_all_metrics
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

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

datasets_collapsed = DATASET_REGISTRY.get('VesselDataset')(path_dict, collapse=True)
train_dataloader_collapsed = DataLoader(datasets_collapsed['train'], batch_size=4, shuffle=True)

model = ARCH_REGISTRY.get('FRNet')(in_channels=1, out_channels=1)
batch = next(iter(train_dataloader_collapsed))
images, labels, names = batch

# Run the model
model.eval()
with torch.no_grad():
    outputs = model(images)

outputs = F.sigmoid(outputs)
outputs = (outputs > 0.5).float()

print(f"  Input shape: {images.shape}")
print(f"  Output shape: {outputs.shape}")
print(f"  Label shape: {labels.shape}")

# Calculate metrics
metrics = calculate_all_metrics(labels, outputs)

# Print metrics
print("\nCalculated Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

# Plot images, labels, and outputs
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
for i in range(4):
    axes[i, 0].imshow(images[i, 0].numpy(), cmap='gray')
    axes[i, 0].set_title(f"Image: {names[i]}")
    axes[i, 1].imshow(labels[i, 0].numpy(), cmap='gray')
    axes[i, 1].set_title("Original Label")
    axes[i, 2].imshow(outputs[i, 0].numpy(), cmap='gray')
    axes[i, 2].set_title("Model Output")

    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()

# Save the plot in the results folder
os.makedirs("results", exist_ok=True)
plt.savefig("results/frnet_test_results.png")
plt.close(fig)