import time
import random
import numpy as np
import torch
from dataset import VideoDataset, VideoStreamDataset, preprocess_video
from utils.train_utils import *
from train_vivit import create_balanced_subset
import h5py
from tqdm import tqdm
from torchvision.models import vit_b_16
from einops import rearrange, repeat
import torch.nn as nn
import decord

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


@torch.no_grad()
def save_embeddings_to_h5(data_dir, output_file, encoder, batch_size=16, step=25, normalize=True):
    """
    Precompute per-frame embeddings and save them into .h5 file.

    h5 file maps frame indices to embeddings: {video_name: {index: [embedding vector]}}

    Args:
        data_dir: VideoStreamDataset instance
        output_file (str): directory where .h5 files will be stored
        encoder (nn.Module): encoder producing (1, T, D) embeddings
        batch_size (int): frames processed per batch
        step (int): step size in video (25 for 1-second step for 25 FPS video)
        normalize (bool): image normalization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()

    video_list = [os.path.join(data_dir, v) for v in os.listdir(data_dir) if v.lower().endswith('.mp4')]
    print(f"Found {len(video_list)} unique videos.")

    with h5py.File(output_file, 'w') as h5f:
        for video_path in tqdm(video_list, desc="Saving embeddings"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            # Skip if already processed
            if video_name in h5f:
                continue

            try:
                vr = decord.VideoReader(video_path, num_threads=4)
            except Exception as e:
                print(f"[WARN] Skipping {video_path}: {e}")
                continue  # skip this file

            num_frames = len(vr)

            grp = h5f.create_group(video_name)

            frame_indices = list(range(0, num_frames, step))

            for i in range(0, len(frame_indices), batch_size):
                batch_indices = frame_indices[i:i+batch_size]
                frames = vr.get_batch(batch_indices).asnumpy()

                frames = preprocess_video(frames, normalize=normalize)

                frames = frames.to(device)
                frames = rearrange(frames, 'b h w c -> b c h w')
                emb = encoder(frames)  # (B, D)
                emb = emb.squeeze(0).cpu().numpy()

                for idx, vec in zip(batch_indices, emb):
                    grp.create_dataset(str(idx), data=vec)


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    random.seed(config['training']['seed'])
    data_config = config['data']

    # save_dataset(data_config=data_config, dataset_split='train', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])
    # save_dataset(data_config=data_config, dataset_split='val', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])
    # save_dataset(data_config=data_config, dataset_split='test', output_path='dataset_split/', balanced_dataset=config['training']['balance_dataset'])
    # data_dir = "/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/"
    data_dir = "/storage/plzen4-ntis/projects/korpusy_cv/RAVDAI/data-240p/"
    encoder = vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    encoder.heads = nn.Identity()
    # save_embeddings_to_h5(data_dir, "/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/embeddings.h5", encoder, batch_size=16, step=25, normalize=data_config['normalize'])
    save_embeddings_to_h5(data_dir, "/storage/plzen4-ntis/projects/korpusy_cv/RAVDAI/embeddings.h5", encoder, batch_size=16, step=25, normalize=data_config['normalize'])

