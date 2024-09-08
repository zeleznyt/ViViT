import av
import numpy as np
import torch
import xml.etree.ElementTree as ET
import sys
import os
import json
import cv2
from torch.utils.data import Dataset, DataLoader
import decord
from decord import VideoReader



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
    def __init__(self, meta_file, classes, frame_sample_rate=1, min_sequence_length=2, max_sequence_length=16,
                 input_fps=25, step=1000, video_decoder='decord'):
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
        self.video_decoder = video_decoder
        self.step = step
        self.original_frame_step = int(1 / self.input_fps * self.step)

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
                if annotation[2] not in self.classes:
                    continue
                start_frame = np.ceil(annotation[0] / self.step) * self.step
                end_frame = np.floor(annotation[1] / self.step) * self.step
                for i in range(int((end_frame - start_frame) / (self.max_sequence_length * self.step))):
                    indexes = (int(start_frame + (i * self.max_sequence_length * self.step)),
                               int(start_frame + (i + 1) * self.max_sequence_length * self.step-self.step))
                    self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                    # print([annotation_file['video'], indexes, annotation[2]])

                last_len = (end_frame - start_frame) % (self.max_sequence_length * self.step)
                indexes = (int(end_frame - last_len), int(end_frame))
                if last_len > self.min_sequence_length * self.step:
                    self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                    # print([annotation_file['video'], indexes, annotation[2]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index: Index of sample to be fetched.
        """
        indices = sample_frame_indices(self.data[index][1], self.step)
        if self.video_decoder == 'pyav':
            video_path = self.data[index][0]
            video = read_video_pyav(container=self.video_handler[video_path], indices=indices)
        elif self.video_decoder == 'decord':
            video_path = self.data[index][0]
            decord_vr = decord.VideoReader(video_path, num_threads=1)
            video = list(decord_vr.get_batch([indices]).asnumpy())
        else:
            print('Unknown video decoder. Must be one of ["pyav", "decord"]')
            return None

        pad_len = self.max_sequence_length - len(video)
        video_padded = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=-1)

        padding_mask = [False] * len(video) + [True] * pad_len

        # image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        # x = image_processor(list(video), return_tensors="pt")
        video_padded = preprocess_video(video_padded)
        x = np.stack(video_padded)
        y = self.data[index][2]

        return x, y, np.array(padding_mask)
