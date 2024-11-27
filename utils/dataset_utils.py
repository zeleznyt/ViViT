import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a PyTorch dataset into train, validation, and test subsets with stratified sampling.

    Args:
        dataset: The PyTorch dataset object.
        train_ratio: Fraction of data to use for training.
        val_ratio: Fraction of data to use for validation.
        test_ratio: Fraction of data to use for testing.
        seed: Random seed for reproducibility.

    Returns:
        train_dataset, val_dataset, test_dataset: Subsets of the original dataset.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."
    original_data = dataset.data
    targets = [i[2] for i in original_data] # Get class labels

    train_idx, remaining_idx = train_test_split(
        range(len(targets)),
        test_size=val_ratio + test_ratio,
        stratify=targets,
        random_state=seed
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=test_ratio / (val_ratio + test_ratio),  # Adjust the split proportionally
        stratify=[targets[i] for i in remaining_idx],
        random_state=seed
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset


def create_split(original_dataset, output_path='dataset_split', dataset_name='VideoStreamDataset',
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    print(f'Creating dataset {dataset_name} train + val + test split...')
    train_dataset, val_dataset, test_dataset = stratified_split(original_dataset,
                                                                train_ratio=train_ratio, val_ratio=val_ratio,
                                                                test_ratio=test_ratio, seed=seed)
    print(f'Indexes created')
    os.makedirs(output_path, exist_ok=True)

    train_data = [original_dataset.data[i] for i in train_dataset.indices]
    train_data = [convert_to_serializable(item) for item in train_data]
    with open(os.path.join(output_path, dataset_name+'.train.json'), 'w') as ftr:
        json.dump(train_data, ftr)
    print(f"Train data saved to {os.path.join(output_path, dataset_name+'.train.json')}")


    val_data = [original_dataset.data[i] for i in val_dataset.indices]
    val_data = [convert_to_serializable(item) for item in val_data]
    with open(os.path.join(output_path, dataset_name+'.val.json'), 'w') as fva:
        json.dump(val_data, fva)
    print(f"Validation data saved to {os.path.join(output_path, dataset_name+'.val.json')}")


    test_data = [original_dataset.data[i] for i in test_dataset.indices]
    test_data = [convert_to_serializable(item) for item in test_data]
    with open(os.path.join(output_path, dataset_name+'.test.json'), 'w') as fte:
        json.dump(test_data, fte)
    print(f"Test data saved to {os.path.join(output_path, dataset_name+'.test.json')}")