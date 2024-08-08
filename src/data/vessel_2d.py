import autorootcwd
import os, torch
import random

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from monai.data import PILReader

from monai.transforms import (
    LoadImage,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    Compose,
    SpatialPadd,
    CenterSpatialCropd,
)
from src.utils.registry import DATASET_REGISTRY
import numpy as np
import matplotlib.pyplot as plt

@DATASET_REGISTRY.register()
def VesselDataset(path_dict, collapse=True):
    all_datasets = {}
    for dataset_name, path in path_dict.items():
        if dataset_name == "ROSSA":
            all_datasets[dataset_name] = {
                "train": ConcatDataset([
                    SegmentationDataset(os.path.join(path, "train_manual"), mode="train"),
                    SegmentationDataset(os.path.join(path, "train_sam"), mode="train")
                ]),
                "val": SegmentationDataset(os.path.join(path, "val"), mode="val"),
                "test": SegmentationDataset(os.path.join(path, "test"), mode="test"),
            }
        else:
            all_datasets[dataset_name] = {
                "train": SegmentationDataset(os.path.join(path, "train"), mode="train"),
                "val": SegmentationDataset(os.path.join(path, "val"), mode="val"),
                "test": SegmentationDataset(os.path.join(path, "test"), mode="test"),
            }

    if collapse:
        combined_datasets = {"train": [], "val": [], "test": []}
        for dataset in all_datasets.values():
            for split in ["train", "val", "test"]:
                combined_datasets[split].append(dataset[split])

        for split in ["train", "val", "test"]:
            all_datasets[split] = ConcatDataset(combined_datasets[split])

        # Remove individual dataset entries
        all_datasets = {k: v for k, v in all_datasets.items() if k in ["train", "val", "test"]}
    else:
        # Collapse only train and val datasets
        combined_train = ConcatDataset([dataset["train"] for dataset in all_datasets.values()])
        combined_val = ConcatDataset([dataset["val"] for dataset in all_datasets.values()])
        
        # Keep individual test datasets
        test_datasets = {name: dataset["test"] for name, dataset in all_datasets.items()}
        
        all_datasets = {"train": combined_train, "val": combined_val, "test": test_datasets}

    return all_datasets


class SegmentationDataset(Dataset):
    def __init__(self, ls_path_dataset, start=0, end=1, mode="train") -> None:
        super().__init__()

        if not isinstance(ls_path_dataset, list):
            ls_path_dataset = [ls_path_dataset]

        self.ls_item = []
        for path_dataset in ls_path_dataset:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")

            ls_file = os.listdir(path_dir_image)

            dataset_name = os.path.basename(os.path.dirname(path_dataset))
            split_name = os.path.basename(path_dataset)

            for name in ls_file:
                path_image = os.path.join(path_dir_image, name)
                path_label = os.path.join(path_dir_label, name)
                assert os.path.exists(path_image)
                assert os.path.exists(path_label)
                self.ls_item.append(
                    {
                        "name": f"{dataset_name}/{split_name}/image/{name}",
                        "image": path_image,
                        "label": path_label,
                    }
                )

        random.seed(0)
        random.shuffle(self.ls_item)
        start = int(start * len(self.ls_item))
        end = int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start:end]

        # Set up Monai transforms
        self.image_loader = LoadImage(reader=PILReader(), image_only=True)

        self.post_transforms = Compose(
            [
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
                SpatialPadd(
                    keys=["image", "label"], spatial_size=(416, 416)
                ),  # Resize to a fixed size
            ]
        )

        # Add augmentation transforms for training
        self.aug_transforms = Compose(
            [
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256)),
            ]
        )
        self.mode = mode

    def __len__(self):
        return len(self.ls_item)

    def __getitem__(self, index):
        index = index % len(self)
        item = self.ls_item[index]

        name = item["name"]
        image_path = item["image"]
        label_path = item["label"]

        # Load image and label using Monai's PILReader
        image = self.image_loader(image_path)
        label = self.image_loader(label_path)

        data = {"image": image, "label": label}

        data = self.post_transforms(data)

        if self.mode == "train":
            data = self.aug_transforms(data)

        image, label = data["image"], data["label"]

        return image, label, name  # Changed order here
    

if __name__ == "__main__":
    # Example usage
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
    datasets_collapsed = VesselDataset(path_dict, collapse=True)
    print("Collapsed datasets:")
    for split, dataset in datasets_collapsed.items():
        print(f"{split}: {len(dataset)} samples")
    
    # Test with collapse=False
    datasets_uncollapsed = VesselDataset(path_dict, collapse=False)
    print("\nUncollapsed datasets:")
    print(f"Train: {len(datasets_uncollapsed['train'])} samples")
    print(f"Val: {len(datasets_uncollapsed['val'])} samples")
    print("Test:")
    for name, dataset in datasets_uncollapsed['test'].items():
        print(f"  {name}: {len(dataset)} samples")

    # Test a single item from the training set
    train_dataset = datasets_collapsed['train']
    image, label, name = train_dataset[0]  # Changed order here
    print(f"\nSample item - Name: {name}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")

    # Create and test DataLoader for collapsed dataset
    print("\nTesting DataLoader for collapsed dataset:")
    train_dataloader_collapsed = DataLoader(datasets_collapsed['train'], batch_size=4, shuffle=True)
    batch = next(iter(train_dataloader_collapsed))
    print(f"Batch size: {len(batch)}")
    print(f"Images shape: {batch[0].shape}")  # Changed order here
    print(f"Labels shape: {batch[1].shape}")  # Changed order here
    print(f"Names shape: {len(batch[2])}")  # Changed order here

    # Create and test DataLoader for uncollapsed dataset
    print("\nTesting DataLoader for uncollapsed dataset:")
    train_dataloader_uncollapsed = DataLoader(datasets_uncollapsed['train'], batch_size=4, shuffle=True)
    batch = next(iter(train_dataloader_uncollapsed))
    print(f"Batch size: {len(batch)}")
    print(f"Images shape: {batch[0].shape}")  # Changed order here
    print(f"Labels shape: {batch[1].shape}")  # Changed order here
    print(f"Names shape: {len(batch[2])}")  # Changed order here