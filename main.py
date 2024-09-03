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
import yaml

np.random.seed(0)

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return [x.to_ndarray(format="rgb24") for x in frames]


def sample_frame_indices(indexes, sample_rate):
    start_idx = int(indexes[0] / sample_rate)
    end_idx = int(indexes[1] / sample_rate)
    indices = np.linspace(start_idx, end_idx, num=int(end_idx - start_idx + 1))
    return indices


def get_eaf(eaf_file: str):
    """
    Read an EAF file
    :param eaf_file: path to an EAF file
    :return: list of annotations (start_time, end_time, class)
    """

    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # Get time in milliseconds
    time_slot_list = [time_slot.attrib for time_slot in root[1]]
    time_order = {slot['TIME_SLOT_ID']: int(slot['TIME_VALUE']) for slot in time_slot_list}

    # Get annotations
    annotation_list = [annotation for annotation in root[2]]
    annotations = []
    for annotation in annotation_list:
        time_slots = annotation[0].attrib['TIME_SLOT_REF1'], annotation[0].attrib['TIME_SLOT_REF2']
        value = annotation[0][0].text
        annotations.append((time_order[time_slots[0]], time_order[time_slots[1]], value))
    return annotations


def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image


def preprocess_video(video):
    result = []
    for image in video:
        result.append(preprocess_image(image))
    return result


class VideoDataset(Dataset):
    def __init__(self, meta_file, classes, frame_sample_rate=1, min_sequence_length=2, max_sequence_length=16, input_fps=25, step=1000, max_len=None):
        """
        Args:
            meta_file (`str`): Path to the metafile containing paths to video and annotation files
            classes (`list`): List of classes
            frame_sample_rate (`int`): Frame sampling
            min_sequence_length (`int`): Minimum number of frames in sequence
            max_sequence_length (`int`): Maximum number of frames in sequence
            input_fps (`int`): Frame sampling of input video
            step (`int`): Number of annotation time steps in one second (1000 for milliseconds)
        """
        self.meta_file = meta_file
        self.classes = classes
        self.data = []
        self.frame_sample_rate = frame_sample_rate
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.input_fps = input_fps
        self.step = step
        self.original_frame_step = int(1 / self.input_fps * self.step)
        if max_len is not None:  # TODO: this is just for debug
            self.max_len = max_len
        else:
            self.max_len = np.inf

        with open(self.meta_file, 'r') as f:
            self.meta_data = json.load(f)

        self.video_handler = {}

        for annotation_file in self.meta_data:
            video_path = annotation_file['video']
            if video_path not in self.video_handler.keys():
                self.video_handler[video_path] = av.open(video_path)

            annotation_list = get_eaf(annotation_file['annotation'])
            # print(annotation_list)

            for annotation in annotation_list:
                if len(self.data) >= self.max_len:  # TODO: just for debug
                    break
                if annotation[2] not in self.classes:
                    continue
                start_frame = np.ceil(annotation[0] / self.step) * self.step
                end_frame = np.floor(annotation[1] / self.step) * self.step
                for i in range(int((end_frame - start_frame) / (self.max_sequence_length * self.step))):
                    indexes = (int(start_frame + (i * self.max_sequence_length * self.step)),
                               int(start_frame + (i + 1) * self.max_sequence_length * self.step-self.step))
                    self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                    # print([annotation_file['video'], indexes, annotation[2]])
                    if len(self.data) >= self.max_len:  # TODO: just for debug
                        break

                if len(self.data) >= self.max_len:  # TODO: just for debug
                    break
                last_len = (end_frame - start_frame) % (self.max_sequence_length * self.step)
                indexes = (int(end_frame - last_len), int(end_frame))
                if last_len > self.min_sequence_length * self.step:
                    self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                    # print([annotation_file['video'], indexes, annotation[2]])
        # self.data = self.data[1:]  # TODO: remove

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index: Index of sample to be fetched.
        """
        indices = sample_frame_indices(self.data[index][1], self.step)
        video = read_video_pyav(container=self.video_handler[self.data[index][0]], indices=indices)
        # x = skimage.transform.resize(video, (16, 224, 224, 3), anti_aliasing=True)

        pad_len = self.max_sequence_length - len(video)
        video_padded = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=-1)

        padding_mask = [False] * len(video) + [True] * pad_len

        # image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        # x = image_processor(list(video), return_tensors="pt")
        video_padded = preprocess_video(video_padded)
        x = np.stack(video_padded)
        y = self.data[index][2]

        return x, y, np.array(padding_mask)


def train_epoch(model, optimizer, data_loader, loss_history, loss_func, device):
    # src: https://github.com/tristandb8/ViViT-pytorch/blob/develop/utils.py
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target, padding_mask) in enumerate(data_loader):
        optimizer.zero_grad()
        # x = data.cuda()
        # data = rearrange(x, 'b p h w c -> b p c h w').cuda()
        # target = target.type(torch.LongTensor).cuda()
        x = data.to(device)
        data = rearrange(x, 'b p h w c -> b p c h w')
        target = target.type(torch.LongTensor).to(device)
        # x = data
        # data = rearrange(x, 'b p h w c -> b p c h w')
        # target = target.type(torch.LongTensor)

        # print('train target:')
        # print(target)
        # pred = model(data.float(), padding_mask=padding_mask)
        pred = model(data.float())
        # print('pred.shape')
        # print(pred.shape)
        # output = F.log_softmax(pred, dim=1)
        # print('output.shape')
        # print(output.shape)
        # loss = F.nll_loss(output, target)
        # output = model(data.float())
        # print('train output:')
        # print(output)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


if __name__ == "__main__":

    config_path = 'config.yaml'
    config = load_config(config_path)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)

    dataset = VideoDataset(data_config['meta_file'], CLASSES, max_sequence_length=data_config['num_frames'])
    train_dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
                                  drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75])

    train_loss_history, test_loss_history = [], []

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_dataloader, train_loss_history, criterion, device)
        # model, train_dataloader, criterion, optimizer = train_epoch(model, optimizer, train_dataloader, epoch, criterion)
        lr_sched.step()

        if epoch % 5 == 0:
            if epoch > 0:
                print("ENTERING EVALUATION")
                # train_accuracy(model,epoch, logFile, trainKitchen)
                # validate(model,epoch, logFile, testKitchen)
                print("Evaluation not implemented yet")

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
