import argparse
import yaml
import wandb
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for ViViT model")

    # Argument for config file
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    # Parse arguments and return them
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def init_wandb(project_name, config):
    system_config = ['PBS_JOBID']
    config['system'] = {}
    for variable in system_config:
        if variable in os.environ.keys():
            config['system'][variable] = os.environ[variable]
    wandb.init(project=project_name, config=config)