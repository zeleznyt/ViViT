import numpy as np
import json
import time
import random
from dataset import VideoDataset, VideoStreamDataset
from utils.train_utils import *
from train_vivit import create_balanced_subset

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def convert_types(o):
    if isinstance(o, np.integer):  # Handles int64, int32, etc.
        return int(o)


def save_dataset(data_config, dataset_split='train', output_path='dataset/', balanced_dataset=False):
    assert data_config['dataset_type'] in ['one_class', 'stream'], f'Dataset type {data_config["dataset_type"]} not supported'
    assert dataset_split in ['train', 'val', 'test'], f'Dataset split {dataset_split} not supported'
    dataset_type = data_config['dataset_type']

    # Create dataset
    start = time.time()
    print('Loading dataset...')

    if dataset_type == 'one_class':
        dataset = VideoDataset(data_config[f'{dataset_split}_meta_file'], CLASSES,
                               load_from_json=None,
                               frame_sample_rate=data_config['frame_sample_rate'],
                               min_sequence_length=data_config['min_sequence_length'],
                               max_sequence_length=data_config['max_sequence_length'],
                               num_threads=data_config['decord_num_threads'],
                               normalize=data_config['normalize'],)
    elif dataset_type == 'stream':
        dataset = VideoStreamDataset(data_config[f'{dataset_split}_meta_file'], CLASSES,
                                     load_from_json=None,
                                     frame_sample_rate=data_config['frame_sample_rate'],
                                     context_size=data_config['context_size'],
                                     overlap=data_config['overlap'],
                                     max_empty_frames=data_config['max_empty_frames'],
                                     num_threads=data_config['decord_num_threads'],
                                     normalize=data_config['normalize'],)
    end = time.time()
    print('Dataset "{}" successfully loaded in {} seconds.'.format(dataset_type, end - start))

    if balanced_dataset:
        # Saving part of the code must be here for Subset
        start = time.time()
        print(f'Balancing {dataset_split} dataset...')
        balanced_dataset = create_balanced_subset(dataset)
        indexes = balanced_dataset.indices
        data = balanced_dataset.dataset.data
        new_data = [data[i] for i in indexes]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, f'balanced_{dataset_type}_{dataset_split}_data.json'), 'w') as _f:
            json.dump(new_data, _f, default=convert_types)
        end = time.time()
        print('Dataset "{}" successfully balanced in {} seconds.'.format(dataset_type, end - start))
    else:
        return dataset.save_dataset_to_json(os.path.join(output_path, f'{dataset_type}_{dataset_split}_data.json'))

if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    random.seed(config['training']['seed'])
    data_config = config['data']

    save_dataset(data_config=data_config, dataset_split='train', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])
    save_dataset(data_config=data_config, dataset_split='val', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])
    save_dataset(data_config=data_config, dataset_split='test', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])


