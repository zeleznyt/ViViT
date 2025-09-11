import argparse
import yaml
import wandb
import os
import json
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for ViViT model")

    # Argument for config file
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--demo_video', type=str, required=False, help='Path to the demo video')
    parser.add_argument('--demo_video_metadata', type=str, required=False, help='Path to the metadata file with demo videos')
    parser.add_argument('--demo_output_path', type=str, default='result/', help='Path to save the result demo files')
    parser.add_argument('--verbose', action='store_true', help='Show debug info')

    # Parse arguments and return them
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def init_wandb(project_name, config, name=None):
    system_config = ['PBS_JOBID']
    config['system'] = {}
    for variable in system_config:
        if variable in os.environ.keys():
            config['system'][variable] = os.environ[variable]
    wandb.init(project=project_name, config=config, name=name)


def get_class_weights(data_config, classes):
    """
    Return weights for each class based on their count in train json
    :param data_config: data config
    :return: weights for each class
    """
    train_data_path = data_config['train_json']
    if train_data_path and os.path.exists(train_data_path):
        with open(train_data_path) as f:
            data = json.load(f)
        # Extract labels
        labels = [item['label'] for item in data]
        # Count occurrences
        counts = Counter(labels)
    else:
        print('No train json file {}. Using default values.'.format(train_data_path))
        counts = {'indoor': 68666,
         'zábava': 68666,
         'studio': 51664,
         'outdoor': 49213,
         'reklama': 32439,
         'upoutávka': 15830,
         'grafika': 13361,
         'předěl': 12031}

    # Total samples and num classes
    total = sum(counts.values())
    num_classes = len(classes)

    # Compute inverse frequency weights (aligned with CLASSES order)
    weights = [total / (num_classes * counts[c]) for c in classes]
    return weights

