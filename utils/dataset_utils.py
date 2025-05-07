import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils.train_utils import *
from dataset import VideoDataset, VideoStreamDataset

def get_video_filename(q_file, video_path):
    for video in os.listdir(video_path):
        if os.path.isfile(os.path.join(video_path, video)):
            file_name = video.replace('.mp4', '')
            if file_name == q_file:
                return os.path.join(video_path, video)
    else:
        return None


def create_metadata_file(video_path, eaf_path, output_file='metadata.json'):
    """
    Create metadata file from list of videos and eaf files
    :param video_path: Path to videos
    :param eaf_path: Path to eaf files
    :param output_file: Output json file with metadata
    :return: json contents
    """
    annot_list = []
    id = 0
    for annotator in os.listdir(eaf_path):
        if os.path.isdir(os.path.join(eaf_path, annotator)):
            ann = os.path.join(eaf_path, annotator)
            for batch in os.listdir(ann):
                f_b = os.path.join(ann, batch)
                for file in os.listdir(f_b):
                    q_file = file.replace('.eaf', '')
                    video_filename = get_video_filename(q_file, video_path)
                    if video_filename is not None:
                        print(file, video_filename, annotator, batch)
                        annot_list.append(
                            {'id': id, 'name': q_file, 'video': video_filename, 'annotation': os.path.join(f_b, file),
                             'annotator': annotator, 'batch': batch})
                        id += 1

    with open(output_file, 'w') as f:
        json.dump(annot_list, f)
    print('Metadata file created.')
    return annot_list


def create_metadata_full_split(video_path, eaf_path, output_dir='/metadata'):
    """
    Create train/val/test metadata files from list of videos and eaf files
    :param video_path: Path to videos
    :param eaf_path: Path to eaf files
    :param output_dir: Output directory of the metadata split
    """
    full_metadata_file=os.path.join(output_dir, 'metadata.json')
    create_metadata_file(video_path, eaf_path, output_file=full_metadata_file)
    split_by_RAVDAI_TVchannel(original_metadata=full_metadata_file)
    print(f'Metadata file created in {output_dir}.')


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


def split_by_RAVDAI_TVchannel(original_metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split metafile into train, validation, and test subset metafiles with unified sampling by the TV channels.

    Args:
        original_metadata: Full metadata (annotation) file to be split.
        train_ratio: Fraction of data to use for training.
        val_ratio: Fraction of data to use for validation.
        test_ratio: Fraction of data to use for testing.
        seed: Random seed for reproducibility.

    Returns:
        train_dataset, val_dataset, test_dataset: Metafile subsets of the original metadata file.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    with open(original_metadata, 'r') as metaf:
        metadata = json.load(metaf)

    base_path = os.path.dirname(original_metadata)
    metafile_name = os.path.basename(original_metadata)

    annotation_basenames = [i['name'].split(' ')[0] for i in metadata]

    train_idx, remaining_idx = train_test_split(
        range(len(annotation_basenames)),
        test_size=val_ratio + test_ratio,
        stratify=annotation_basenames,
        random_state=seed
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=test_ratio / (val_ratio + test_ratio),  # Adjust the split proportionally
        stratify=[annotation_basenames[i] for i in remaining_idx],
        random_state=seed
    )

    train_metadata = [metadata[i] for i in train_idx]
    val_metadata = [metadata[i] for i in val_idx]
    test_metadata = [metadata[i] for i in test_idx]

    with open(os.path.join(base_path, 'train_'+metafile_name), 'w') as ftr:
        json.dump(train_metadata, ftr, indent=4)
    with open(os.path.join(base_path, 'val_'+metafile_name), 'w') as fva:
        json.dump(val_metadata, fva, indent=4)
    with open(os.path.join(base_path, 'test_'+metafile_name), 'w') as fte:
        json.dump(test_metadata, fte, indent=4)
    print(f"Train data saved to {base_path}.")


if __name__ == '__main__':
    split_by_RAVDAI_TVchannel('/media/zeleznyt/DATA/data/RAVDAI/new_annot/annotations.json')
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    data_config = config['data']

    # Create dataset
    print('Loading dataset...')
    classes = ['studio', 'indoor', 'outdoor']
    assert data_config['dataset_type'] in ['one_class',
                                           'stream'], f'Dataset type {data_config["dataset_type"]} not supported'
    if data_config['dataset_type'] == 'one_class':
        dataset = VideoDataset(data_config['meta_file'], classes,
                                     load_from_json=data_config['train_json'],
                                     frame_sample_rate=data_config['frame_sample_rate'],
                                     min_sequence_length=data_config['min_sequence_length'],
                                     max_sequence_length=data_config['max_sequence_length'],
                                     video_decoder=data_config['video_decoder'], )
    elif data_config['dataset_type'] == 'stream':
        dataset = VideoStreamDataset(data_config['meta_file'], classes,
                                           load_from_json=data_config['train_json'],
                                           frame_sample_rate=data_config['frame_sample_rate'],
                                           context_size=data_config['context_size'],
                                           overlap=data_config['overlap'],
                                           max_empty_frames=data_config['max_empty_frames'],
                                           video_decoder=data_config['video_decoder'], )
    print('Dataset "{}" successfully loaded.'.format(data_config['dataset_type']))
    create_split(dataset, output_path='dataset_split', dataset_name=data_config['dataset_type'],)