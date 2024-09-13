import argparse
import av
import numpy as np
import torch
import xml.etree.ElementTree as ET
import sys
import os
import json
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video.resnet import model_urls
from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download
import cv2
from vivit import ViViT
from dataset import VideoDataset
from dataset import visualize_frames
import yaml
import wandb
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from torch.utils.data import Subset

np.random.seed(0)

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']

def train_epoch(epoch, model, optimizer, data_loader, loss_history, loss_func, device, checkpoint_save_dir, log_step=100, eval_step=-1, save_step=-1, report_to=None):
    # src: https://github.com/tristandb8/ViViT-pytorch/blob/develop/utils.py
    total_samples = len(data_loader.dataset)
    model.train()

    start_time = time.time()
    for i, (data, target, padding_mask) in enumerate(data_loader):
        # Use this to visualize th data
        # visualize_frames(data.numpy()[0], CLASSES[target[0].numpy()])
        optimizer.zero_grad()
        x = data.to(device)
        padding_mask = padding_mask.to(device)
        data = rearrange(x, 'b p h w c -> b p c h w')
        target = target.type(torch.LongTensor).to(device)

        pred = model(data.float(), padding_mask)

        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        lr_sched.step()

        end_time = time.time()

        if i % log_step == 0 or i == len(data_loader) - 1:
            # Log to wandb
            if report_to == 'wandb':
                wandb.log({"train_loss": loss.item(), "time_per_iteration": (end_time - start_time)/log_step, "epoch": epoch,
                           "learning_rate": optimizer.param_groups[0]['lr']})

            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            start_time = time.time()

        if (i % eval_step == 0 or i == len(data_loader) - 1) and eval_step != -1:
            print('Eval here. Not implemented yet')

        if (i % save_step == 0 or i == len(data_loader) - 1) and save_step != -1:
            model_path = os.path.join(checkpoint_save_dir, 'model_{}-{}.pt'.format(epoch, i))
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print('Model successfully saved to {}'.format(model_path))


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


def dataset_distribution(dataset, plot=False):
    class_counts = {}
    print('Counting classes...')
    for data in dataset:
        _, class_label, _ = data

        if class_label not in class_counts:
            class_counts[class_label] = 0
        class_counts[class_label] += 1

    print("Number of samples per class:")
    for class_label, count in class_counts.items():
        print(f"Class {CLASSES[class_label]}: {count} samples")

    if plot:
        class_labels = list(class_counts.keys())
        class_labels = [CLASSES[c] for c in class_labels]
        counts = list(class_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, counts)

        plt.xlabel('Class Labels')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return class_counts


def create_balanced_subset(dataset, n_of_instances=-1):
    class_indices = defaultdict(list)
    for idx, data in enumerate(dataset):
        _, class_label, _ = data
        class_indices[class_label].append(idx)
    min_class_count = min(len(indices) for indices in class_indices.values())
    if n_of_instances < 0 or n_of_instances > min_class_count:
        n_of_instances = min_class_count
    balanced_indices = []
    for class_label, indices in class_indices.items():
        balanced_indices.extend(random.sample(indices, n_of_instances))
    balanced_subset = Subset(dataset, balanced_indices)
    print('Balanced subset created with {} instances for each class.'.format(n_of_instances))
    return balanced_subset

if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    warmup_epochs = train_config['warmup_epochs']
    learning_rate = train_config['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = ViViT(model_config).to(device)
    # Move non-trainable mask to the device
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    print('Loading dataset...')
    dataset = VideoDataset(data_config['meta_file'], CLASSES, frame_sample_rate=data_config['frame_sample_rate'],
                           min_sequence_length=data_config['min_sequence_length'],
                           max_sequence_length=data_config['max_sequence_length'],
                           video_decoder=data_config['video_decoder'],)
    if train_config['balance_dataset']:
        dataset = create_balanced_subset(dataset, 10)
    train_dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
                                  drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])
    print('Dataset successfully loaded.')

    # dataset_distribution(dataset, True)

    criterion = nn.CrossEntropyLoss()
    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print('Unknown optimizer. Must be one of ["adam"]. Setting "adam" instead...')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = len(train_dataloader)
    if train_config['lr_scheduler'] == 'cosine':
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=(num_epochs - warmup_epochs) * steps_per_epoch)
    elif train_config['lr_scheduler'] == 'constant':
        lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif train_config['lr_scheduler'] == 'multistep':
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75])
    else:
        print('Unknown optimizer. Must be one of ["cosine", "constant", "multistep"]. Setting "constant" instead]...')
        lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer)

    if train_config['warmup_epochs'] > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                                    total_iters=int(warmup_epochs * steps_per_epoch))
        lr_sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_sched],
                                 milestones=[int(warmup_epochs * steps_per_epoch)])

    train_loss_history, test_loss_history = [], []

    model_name = 'ViVit-B_{}x{}'.format(model_config['patch_size'], model_config['tubelet_size'])

    project_name='ViViT'
    if train_config['report_to'] == 'wandb':
        init_wandb(project_name, config)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        train_epoch(epoch, model, optimizer, train_dataloader, train_loss_history, criterion, device,
                    log_step=train_config['log_step'], eval_step=train_config['eval_step'],
                    save_step=train_config['save_step'],
                    checkpoint_save_dir=os.path.join(train_config['checkpoint_save_dir'], model_name),
                    report_to=train_config['report_to'])
        # model, train_dataloader, criterion, optimizer = train_epoch(model, optimizer, train_dataloader, epoch, criterion)

    # test_dataset = VideoDataset('/home/zeleznyt/Documents/Sandbox/vivit/annotations.json', CLASSES, max_sequence_length=16, max_len=10)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = next(iter(train_dataloader))
    test_video = test_data[0]
    # test_video = rearrange(test_video, 'b p h w c -> b p c h w').cuda()
    test_video = rearrange(test_video, 'b p h w c -> b p c h w')
    pred = model(test_video)
    output = F.log_softmax(pred, dim=1)
    pred_class = [CLASSES[v.argmax()] for v in output]
    gt_class = [CLASSES[l] for l in test_data[1]]
    print('GT classes:')
    print(gt_class)
    print('Predicted classes:')
    print(pred_class)
