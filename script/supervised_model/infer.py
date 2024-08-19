import autorootcwd
import src.data
import src.losses
import src.metrics
import src.models

import os
import csv
import click

from src.models.supervised_model import SupervisedModel
from src.utils.registry import DATASET_REGISTRY, ARCH_REGISTRY, LOSS_REGISTRY
from torch.utils.data import DataLoader
from src.utils.trainer import Inferer
from src.metrics.vessel_2d import calculate_all_metrics

import numpy as np

@click.command()
@click.argument('result_folder', type=click.Path(exists=True), required=False)
@click.option('--batch-size', default=8, help='Inference batch size')
@click.option('--interactive', is_flag=True, help='Run in interactive mode')
def main(result_folder, batch_size, interactive):
    if interactive or result_folder is None:
        result_folder = click.prompt('Result folder', type=click.Path(exists=True))
        batch_size = click.prompt('Inference batch size', type=int, default=batch_size)
    elif result_folder is None:
        raise click.UsageError("The 'result_folder' argument is required when not in interactive mode.")

    # Infer dataset, architecture, and loss from the result folder name
    folder_name = os.path.basename(result_folder)
    parts = folder_name.split('_')
    dataset = parts[2]
    arch = parts[0]
    criterion = parts[1]

    path_dict = {
        dataset: f"data/{dataset}"
    }

    datasets_collapsed = DATASET_REGISTRY.get('VesselDataset')(path_dict, collapse=True)
    test_dataset = datasets_collapsed['test']

    print(f"Dataset: {dataset}")
    print(f"Architecture: {arch}")
    print(f"Loss function: {criterion}")
    print(f"Test dataset size: {len(test_dataset)} samples")

    model = SupervisedModel(arch=arch, criterion=criterion, mode='infer')
    model.set_name(folder_name)

    print('TESTING INFERENCE')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    inferer = Inferer(
        model=model,
        test_dataloader=test_dataloader,
        results_folder=os.path.join(result_folder, 'inference'),
        fp16=True,
        checkpoint_path=os.path.join(result_folder, 'model-best-weights.pt')
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
    results_folder = os.path.join(result_folder, 'inference')
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

if __name__ == '__main__':
    main()