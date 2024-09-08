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
import yaml
import wandb
import time

np.random.seed(0)

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']

def train_epoch(epoch, model, optimizer, data_loader, loss_history, loss_func, device, log_step=100, eval_step=-1, save_step=-1):
    # src: https://github.com/tristandb8/ViViT-pytorch/blob/develop/utils.py
    total_samples = len(data_loader.dataset)
    model.train()

    start_time = time.time()
    for i, (data, target, padding_mask) in enumerate(data_loader):
        optimizer.zero_grad()
        x = data.to(device)
        data = rearrange(x, 'b p h w c -> b p c h w')
        target = target.type(torch.LongTensor).to(device)

        pred = model(data.float())

        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

        end_time = time.time()

        if i % log_step == 0 or i == len(data_loader) - 1:
            # Log to wandb
            wandb.log({"train_loss": loss.item(), "time_per_iteration": end_time - start_time, "epoch": epoch,
                       "learning_rate": optimizer.param_groups[0]['lr']})

            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

        if (i % eval_step == 0 or i == len(data_loader) - 1) and eval_step != -1:
            print('Eval here. Not implemented yet')

        if (i % save_step == 0 or i == len(data_loader) - 1) and save_step != -1:
            print('Save here. Not implemented yet')

        start_time = time.time()


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

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = ViViT(model_config).to(device)

    dataset = VideoDataset(data_config['meta_file'], CLASSES, max_sequence_length=data_config['num_frames'])
    train_dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
                                  drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75])

    train_loss_history, test_loss_history = [], []

    project_name='ViViT'
    init_wandb(project_name, config)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        train_epoch(epoch, model, optimizer, train_dataloader, train_loss_history, criterion, device,
                    log_step=train_config['log_step'], eval_step=train_config['eval_step'],
                    save_step=train_config['save_step'])
        # model, train_dataloader, criterion, optimizer = train_epoch(model, optimizer, train_dataloader, epoch, criterion)
        lr_sched.step()

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
