import argparse
import json
import time
from pathlib import Path

import cv2
import decord
import numpy as np
import onnxruntime as ort


CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX inference script for ViViT model")
    parser.add_argument('--model-dir', '--model_dir', dest='model_dir', type=str, required=True,
                        help='Path to the ONNX model directory')
    parser.add_argument('--demo_video', type=str, required=False, help='Path to the demo video')
    parser.add_argument('--cpu', action='store_true', help='Force CPUExecutionProvider')
    return parser.parse_args()


def get_providers(force_cpu=False):
    if force_cpu:
        return ["CPUExecutionProvider"]

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def resize_with_padding(image, target_size=(224, 224), pad_value=(123, 117, 104)):
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
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=tuple(int(v) for v in pad_value)
    )

    return padded_image


def preprocess_frame(frame, image_size=224, pad_value=(123, 117, 104), normalize=True):
    frame = resize_with_padding(frame, target_size=(image_size, image_size), pad_value=pad_value)

    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
    else:
        frame = frame.astype(np.float32)

    frame = np.transpose(frame, (2, 0, 1))
    return np.expand_dims(frame, axis=0).astype(np.float32)


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class ViViTpredictor():
    def __init__(self, model_dir, force_cpu=False):
        self.model_dir = Path(model_dir).expanduser().resolve()
        metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))

        self.classes = metadata.get("classes", CLASSES)
        self.window_length = int(metadata["window_length"])
        self.embed_dim = int(metadata["embed_dim"])
        self.image_size = int(metadata["image_size"])
        self.pad_value = metadata.get("pad_value", [123, 117, 104])
        self.filled = 0
        self.frames = []

        providers = get_providers(force_cpu)

        print('Loading checkpoint...')
        self.spatial_encoder = ort.InferenceSession(
            str(self.model_dir / metadata["spatial_model"]),
            providers=providers,
        )
        self.temporal_encoder = ort.InferenceSession(
            str(self.model_dir / metadata["temporal_model"]),
            providers=providers,
        )

        self.spatial_input_name = self.spatial_encoder.get_inputs()[0].name
        self.temporal_input_names = [input_meta.name for input_meta in self.temporal_encoder.get_inputs()]

        self.embed_frames = np.zeros((self.window_length, self.embed_dim), dtype=np.float32)
        self.padding_mask = np.ones(self.window_length, dtype=np.bool_)

        actual_provider = self.spatial_encoder.get_providers()
        print(f'Model loaded with {actual_provider} providers.')

    def predict(self, frame: np.ndarray) -> tuple[int, list[float]]:
        """
        Predicts a class of the middle frame of the video.
        :param frame: ndarrays frame:
                      Frame shape: height x width x channels (standard decord output)
        :return: predicted_class (int), probs (list):
        """
        if frame is None:
            emb = np.zeros((1, self.embed_dim), dtype=np.float32)
            pad = True
        else:
            self.frames.append(frame)

            preprocessed_frame = preprocess_frame(
                frame,
                image_size=self.image_size,
                pad_value=self.pad_value,
                normalize=True,
            )
            emb = self.spatial_encoder.run(None, {self.spatial_input_name: preprocessed_frame})[0]

            if self.filled < self.window_length:
                self.filled += 1
            else:
                print(self.filled)
                self.frames.pop(0)
            pad = False

        self.embed_frames = np.roll(self.embed_frames, shift=-1, axis=0)
        self.embed_frames[-1] = emb
        self.padding_mask = np.roll(self.padding_mask, shift=-1, axis=0)
        self.padding_mask[-1] = pad

        embed_frames = np.expand_dims(self.embed_frames, axis=0)
        padding_mask = np.expand_dims(self.padding_mask, axis=0)
        pred = self.temporal_encoder.run(
            None,
            {
                self.temporal_input_names[0]: embed_frames,
                self.temporal_input_names[1]: padding_mask,
            },
        )[0]

        probs = softmax(pred)
        predicted_class = np.argmax(probs, axis=1)
        return int(predicted_class.item()), probs.squeeze(0).tolist()

    def class2string(self, cls):
        return self.classes[cls]

    def predict_class_with_probs(self, video):
        cls, probs = self.predict(video)
        predicted_class = self.class2string(cls)
        class_probs = dict(zip(self.classes, probs))
        return predicted_class, class_probs


ViViTONNXPredictor = ViViTpredictor


def frame_to_timestamp(total_seconds):
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(int(total_seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == '__main__':
    # Process args and config
    args = parse_args()
    predictor = ViViTpredictor(args.model_dir, force_cpu=args.cpu)

    # Process random data sample
    # input_video = [np.random.rand(240, 426, 3) for _ in range(80)] # List of ndarrays

    # Process mp4 video
    n_frames_to_load = 250
    fps = 25
    indices = list(range(0, n_frames_to_load * fps, fps))
    video_path = args.demo_video or '/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/ct24 2023-10-02 01.27.34_example.mp4'
    vr = decord.VideoReader(video_path)
    input_video = [vr[i].asnumpy() for i in indices]

    for i, input_frame in enumerate(input_video):
        start_time = time.time()
        result, probs = predictor.predict_class_with_probs(input_frame)
        probs_rounded = {k: f"{v:.2f}" for k, v in probs.items()}
        print(f'Predicted class at {frame_to_timestamp(i-8)}: {result}. All probabilities: {probs_rounded}')
        print('Time elapsed during inference:', time.time() - start_time)

    for i in range(8):
        start_time = time.time()
        result, probs = predictor.predict_class_with_probs(None)
        probs_rounded = {k: f"{v:.2f}" for k, v in probs.items()}
        print(f'Predicted class at {frame_to_timestamp(i-8)}: {result}. All probabilities: {probs_rounded}')
        print('Time elapsed during inference:', time.time() - start_time)
