import numpy as np
import torch
import random
import time
from dataset import VideoDataset, VideoStreamDataset
from utils.train_utils import *
from train_vivit import create_balanced_subset

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def save_dataset(data_config, dataset_split='train', output_path='dataset/', balanced_dataset=False):
    assert data_config['dataset_type'] in ['one_class', 'stream'], f'Dataset type {data_config["dataset_type"]} not supported'
    assert dataset_split in ['train', 'val', 'test'], f'Dataset split {dataset_split} not supported'

    # Create dataset
    start = time.time()
    print('Loading dataset...')
    assert data_config['dataset_type'] in ['one_class', 'stream'], f'Dataset type {data_config["dataset_type"]} not supported'

    if data_config['dataset_type'] == 'one_class':
        dataset = VideoDataset(data_config[f'{dataset_split}_meta_file'], CLASSES,
                               load_from_json=None,
                               frame_sample_rate=data_config['frame_sample_rate'],
                               min_sequence_length=data_config['min_sequence_length'],
                               max_sequence_length=data_config['max_sequence_length'],
                               num_threads=data_config['decord_num_threads'],
                               normalize=data_config['normalize'],)
    elif data_config['dataset_type'] == 'stream':
        dataset = VideoStreamDataset(data_config[f'{dataset_split}_meta_file'], CLASSES,
                                     load_from_json=None,
                                     frame_sample_rate=data_config['frame_sample_rate'],
                                     context_size=data_config['context_size'],
                                     overlap=data_config['context_size'],
                                     max_empty_frames=data_config['max_empty_frames'],
                                     num_threads=data_config['decord_num_threads'],
                                     normalize=data_config['normalize'],)
    end = time.time()
    print('Dataset "{}" successfully loaded in {} seconds.'.format(data_config['dataset_type'], end - start))

    if balanced_dataset:
        start = time.time()
        print('Balancing training dataset...')
        balanced_dataset = create_balanced_subset(dataset)
        indexes = balanced_dataset.indices

        end = time.time()
        print('Dataset "{}" successfully balanced in {} seconds.'.format(data_config['dataset_type'], end - start))

    return dataset.save_dataset_to_json(os.path.join(output_path, f'{dataset_split}_data.json'))

if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    data_config = config['data']

    save_dataset(data_config=data_config, dataset_split='train', output_path='dataset/', balanced_dataset=config['training']['balance_dataset'])
    save_dataset(data_config=data_config, dataset_split='val', output_path='dataset/', balanced_dataset=config['training']['balance_dataset'])
    save_dataset(data_config=data_config, dataset_split='test', output_path='dataset/', balanced_dataset=config['training']['balance_dataset'])


