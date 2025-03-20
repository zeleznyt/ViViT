import av
import numpy as np
import xml.etree.ElementTree as ET
import os
import json
import cv2
from torch.utils.data import Dataset, DataLoader
import decord
import matplotlib.pyplot as plt



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


def resize_with_padding(image, target_size=(224, 224)):
    h, w, _ = image.shape
    target_h, target_w = target_size

    # Calculate the new size that fits within the target size while keeping aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padding
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # Add padding (using a gray color like [123, 117, 104] for ImageNet)
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(123, 117, 104)
    )

    return padded_image


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def preprocess_image(image, target_size=(224, 224), normalize=False):
    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Resize and pad
    image = resize_with_padding(image, target_size)

    # Normalize
    if normalize:
        image = normalize_image(image)
    return image


def preprocess_video(video, normalize=False):
    result = []
    for image in video:
        result.append(preprocess_image(image, normalize=normalize))
    return result


def visualize_frame(image, label=None):
    plt.imshow(image)  # Display image
    if label is not None:
        plt.title(label)
    plt.tight_layout()  # Adjust the layout
    plt.show()


def visualize_frames(video, label=None):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(video):
            break
        ax.imshow(video[i])  # Display image
        ax.axis('off')
    if label is not None:
        plt.suptitle(label)
    plt.tight_layout()  # Adjust the layout
    plt.show()


def debug_indicies(video_path, indicies):
    decord_vr = decord.VideoReader(video_path, num_threads=1)
    video = list(decord_vr.get_batch(indicies).asnumpy())
    visualize_frames(video)


def get_label_on_idx(frame_idx, annotation_list):
    for annotation in annotation_list:
        if annotation[0] <= frame_idx <= annotation[1]:
            return annotation[2]
    else:
        return -1


class VideoDataset(Dataset):
    def __init__(self, meta_file, classes, load_from_json=None, frame_sample_rate=1, min_sequence_length=2,
                 max_sequence_length=16, input_fps=25, step=1000, num_threads=0, normalize=False):
        """
        Args:
            meta_file (`str`): Path to the metafile containing paths to video and annotation files
            classes (`list`): List of classes
            load_from_json (`str`, None): Path to the json file containing exported data by save_dataset_to_json
            frame_sample_rate (`float`): Frame sampling per second. (5 for processing frame each 5th second)
                                         If not result frame index not int, the number is floored (mainly for rate < 1)
            min_sequence_length (`int`): Minimum number of frames in a sequence
            max_sequence_length (`int`): Maximum number of frames in a sequence
            input_fps (`int`): Frame sampling of input video
            step (`int`): Number of annotation time steps in one second (1000 for milliseconds)
            num_threads (`int`): Number of threads for decord
            normalize (`bool`): Normalize the video
        """
        self.meta_file = meta_file
        self.classes = classes
        self.data = []
        self.frame_sample_rate = frame_sample_rate
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.input_fps = input_fps
        self.num_threads = num_threads
        self.step = step
        self.normalize = normalize
        sampling = self.input_fps * self.frame_sample_rate

        video_list = []
        self.meta_data = []
        with open(self.meta_file, 'r') as f:
            meta_data = json.load(f)
            for i, video in enumerate(meta_data):
                if video['video'] not in video_list:
                    video_list.append(video['video'])
                    self.meta_data.append(meta_data[i])

        if load_from_json is not None and os.path.exists(load_from_json):
            print(f'Loading data from {load_from_json}')
            with open(load_from_json, 'r') as f:
                self.data = json.load(f)
        else:
            for annotation_file in self.meta_data:
                annotation_list = get_eaf(annotation_file['annotation'])

                for annotation in annotation_list:
                    if annotation[2] not in self.classes:
                        continue

                    start_frame = int(annotation[0] / self.step * self.input_fps)
                    end_frame = int(annotation[1] / self.step * self.input_fps)
                    for i in range(int(np.ceil((end_frame - start_frame) /
                                               (self.max_sequence_length * self.input_fps * self.frame_sample_rate)))):
                        indexes = list(np.arange(start_frame + i * self.max_sequence_length * sampling,
                                                 start_frame + (i + 1) * self.max_sequence_length * sampling,
                                                 sampling).astype(int))
                        if indexes[-1] <= end_frame and len(indexes) >= self.min_sequence_length:
                            self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                        else:
                            last_len = (end_frame - start_frame) % (self.max_sequence_length * sampling)
                            indexes = list(np.arange(end_frame - last_len, end_frame, sampling).astype(int))
                            if len(indexes) >= self.min_sequence_length:
                                self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index: Index of sample to be fetched.
        """
        indices = self.data[index][1]
        video_path = self.data[index][0]
        decord_vr = decord.VideoReader(video_path, num_threads=self.num_threads)
        video = list(decord_vr.get_batch(indices).asnumpy())

        pad_len = self.max_sequence_length - len(video)
        video_padded = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=-1)

        padding_mask = [False] * len(video) + [True] * pad_len

        video_padded = preprocess_video(video_padded, normalize=self.normalize)
        x = np.stack(video_padded)
        y = self.data[index][2]

        return x, y, np.array(padding_mask)

    def save_dataset_to_json(self, save_dir='dataset/data.json'):
        """
        Saves dataset to json file so it doesn't have to be loaded again.
        Use "load_from_json" to load from json file
        :param save_dir: Path to save dataset.
        """

        def convert_types(o):
            if isinstance(o, np.integer):  # Handles int64, int32, etc.
                return int(o)

        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        with open(save_dir, 'w') as _f:
            json.dump(self.data, _f, default=convert_types)

class VideoStreamDataset(VideoDataset):
    def __init__(self, meta_file, classes, load_from_json=None, frame_sample_rate=1, context_size=8, overlap=2,
                 max_empty_frames=3, input_fps=25, step=1000, num_threads=0, normalize=False):
        """
        Args:
            meta_file (`str`): Path to the metafile containing paths to video and annotation files
            classes (`list`): List of classes
            load_from_json (`str`, None): Path to the json file containing exported data by save_dataset_to_json
            frame_sample_rate (`float`): Frame sampling per second. (5 for processing frame each 5th second)
                                         If not result frame index not int, the number is floored (mainly for rate < 1)
            context_size (`int`): Size of window (data) is 2 x context_size + 1 (center). 17 by default
            overlap (`int`): Overlap of sliding window while creating the dataset
            max_empty_frames (`int`): Maximum number of frames in a sequence that are not labeled with any class. -1 for unlimited
            input_fps (`int`): Frame sampling of input video
            step (`int`): Number of annotation time steps in one second (1000 for milliseconds)
            num_threads (`int`): Number of threads for decord
            normalize (`bool`): Normalize the video
        """

        self.meta_file = meta_file
        self.classes = classes
        self.data = []
        self.frame_sample_rate = frame_sample_rate
        self.context_size = context_size
        self.overlap = overlap
        self.max_empty_frames = max_empty_frames
        if self.max_empty_frames == -1:
            self.max_empty_frames = 9999
        self.input_fps = input_fps
        self.num_threads = num_threads
        self.step = step
        self.max_sequence_length = 2 * context_size + 1
        self.normalize = normalize
        sampling = self.input_fps * self.frame_sample_rate

        video_list = []
        self.meta_data = []
        with open(self.meta_file, 'r') as f:
            meta_data = json.load(f)
            for i, video in enumerate(meta_data):
                if video['video'] not in video_list:
                    video_list.append(video['video'])
                    self.meta_data.append(meta_data[i])

        if load_from_json is not None and os.path.exists(load_from_json):
            print(f'Loading data from {load_from_json}')
            with open(load_from_json, 'r') as f:
                self.data = json.load(f)
        else:
            for annotation_file in self.meta_data:
                annotation_list = get_eaf(annotation_file['annotation'])

                mid_frame = self.context_size * sampling

                while (mid_frame + self.context_size * sampling) / self.input_fps * self.step <= annotation_list[-1][1]:
                    indexes = list(np.arange(mid_frame - (self.context_size * sampling),
                                         mid_frame + ((self.context_size +1) * sampling),
                                         sampling).astype(int))
                    labels = [get_label_on_idx(i / self.input_fps * self.step, annotation_list) for i in indexes]
                    label = labels[self.context_size]
                    # label = get_label_on_idx(indexes[self.context_size] / self.input_fps * self.step, annotation_list)
                    if label == -1 or labels.count(-1) > self.max_empty_frames:
                        mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
                        continue

                    if label not in self.classes:
                        mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
                        continue

                    self.data.append([annotation_file['video'], indexes, self.classes.index(label)])
                    mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
