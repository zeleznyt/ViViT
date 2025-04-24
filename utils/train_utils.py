import argparse
import yaml
import wandb
import os


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