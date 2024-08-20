import autorootcwd
import src.data
import src.losses
import src.metrics
import src.models

import os
import csv

from src.models.segdiff_model import SegDiffModel
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

test_dataset = datasets_collapsed['test']

#printing the arch and criterion available in the registry
print('Available arch in the registry:')
print(ARCH_REGISTRY.keys())

print('Available loss in the registry:')
print(LOSS_REGISTRY.keys())

model = SegDiffModel(arch='SegDiffUnet', criterion='pred_noise', mode='train')
model_folder_name = model.get_name()

print('TESTING INFERENCE')
test_dataloader = DataLoader(test_dataset, batch_size=25, num_workers=8, shuffle=False)

inferer = Inferer(
    model=model,
    test_dataloader=test_dataloader,
    results_folder=os.path.join('results', model_folder_name, 'inference'),
    fp16=True,
    checkpoint_path=os.path.join('results', model_folder_name, 'model-best-weights.pt')
)

outputs, labels = inferer.run()

print(f"Inference complete. Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")


# Calculate metrics for each sample
all_metrics = []
for i in range(outputs.shape[0]):
    metrics = calculate_all_metrics(outputs[i], labels[i])
    all_metrics.append(metrics)

# Prepare data for CSV
csv_data = []
for i, metrics in enumerate(all_metrics):
    row = {'Sample': i+1}
    row.update(metrics)
    csv_data.append(row)

# Calculate mean and std for each metric
mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}

# Add mean and std to csv_data
csv_data.append({'Sample': 'Mean', **mean_metrics})
csv_data.append({'Sample': 'Std', **std_metrics})

# Save results to CSV
results_folder= os.path.join('results', model_folder_name, 'inference')
csv_file_path = os.path.join(results_folder, 'metrics_results.csv')
result_summary_file_path = os.path.join(results_folder, 'final_results.txt')

fieldnames = ['Sample'] + list(all_metrics[0].keys())

with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Results saved to {csv_file_path}")

# Print mean and std
with open(result_summary_file_path, 'w') as f:
    f.write("Mean values:\n")
    print("\nMean values:")
    for key, value in mean_metrics.items():
        line = f"{key}: {value:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\nStandard deviation values:\n")
    print("\nStandard deviation values:")
    for key, value in std_metrics.items():
        line = f"{key}: {value:.4f}"
        print(line)
        f.write(line + "\n")

print(f"Results summary saved to {result_summary_file_path}")