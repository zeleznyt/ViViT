from vivit import ViViT
from dataset import preprocess_video
import torch
from einops import rearrange
import numpy as np
import time
from utils.train_utils import *
from torchvision.models import vit_b_16
import torch.nn as nn
import decord


CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViViTpredictor():
    def __init__(self, config, checkpoint_path, device):
        self.config = config
        self.device = device
        self.classes = CLASSES
        self.window_length = 17
        self.embed_dim = 768
        self.embed_frames = torch.zeros(self.window_length, self.embed_dim, device=self.device)
        self.filled = 0
        self.frames = []

        encoder = vit_b_16(weights=None).to(self.device)
        encoder.heads = nn.Identity()
        self.spatial_encoder = encoder.to(self.device).eval()

        self.temporal_encoder = ViViT(model_config, use_only_embeddings=True).to(self.device).eval()

        num_classes = len(CLASSES)
        model_config['num_classes'] = num_classes

        print('Loading checkpoint...')
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(self.device))
        state_dict = checkpoint["model_state_dict"]

        spatial_state = {}
        for k, v in state_dict.items():
            if k.startswith("spatial_transformer"):
                new_key = k.replace("spatial_transformer.vit.", "")
                spatial_state[new_key] = v

        self.spatial_encoder.load_state_dict(spatial_state, strict=True)

        temporal_state = {}
        for k, v in state_dict.items():
            if k.startswith("temporal_transformer") or k.startswith("classifier"):
                temporal_state[k] = v

        self.temporal_encoder.load_state_dict(temporal_state, strict=True)

        print(f'Model loaded on {self.device} device.')

    def predict(self, frame: np.ndarray) -> tuple[int, list[float]]:
        """
        Predicts a class of the middle frame of the video.
        :param frame: ndarrays frame:
                      Frame shape: height x width x channels (standard decord output)
        :return: predicted_class (int), probs (list):
        """
        self.frames.append(frame)

        preprocessed_frame = preprocess_video(np.expand_dims(frame, axis=0), normalize=True)
        preprocessed_frame = preprocessed_frame.to(self.device)
        preprocessed_frame = rearrange(preprocessed_frame, 'b h w c -> b c h w')
        with torch.no_grad():
            emb = self.spatial_encoder(preprocessed_frame)  # (1, D)

        if self.filled < self.window_length:
            self.filled += 1
        else:
            print(self.filled)
            self.frames.pop(0)

        self.embed_frames = torch.roll(self.embed_frames, shifts=-1, dims=0)
        self.embed_frames[-1] = emb

        # Padding mask
        seq_len = self.filled
        pad_len = self.window_length - seq_len

        # Create a boolean tensor for the padding mask
        padding_mask = torch.cat([
            torch.ones(pad_len, dtype=torch.bool),
            torch.zeros(seq_len, dtype=torch.bool)
        ]).to(self.device)

        embed_frames, padding_mask = [t.unsqueeze(0) for t in (self.embed_frames, padding_mask)]
        with torch.no_grad():
            pred = self.temporal_encoder(embed_frames, padding_mask)

        probs = torch.softmax(pred, dim=1)
        predicted_class = probs.argmax(dim=1)
        return predicted_class.item(), probs.squeeze(0).cpu().tolist()

    def class2string(self, cls):
        return self.classes[cls]

    def predict_class_with_probs(self, video):
        cls, probs = self.predict(video)
        predicted_class = self.class2string(cls)
        class_probs = dict(zip(self.classes, probs))
        return predicted_class, class_probs


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
    config = load_config(args.config)
    model_config = config['model']
    checkpoint_path = config['evaluation']['checkpoint']
    predictor = ViViTpredictor(model_config, checkpoint_path, device=get_device())
    # input_video = [np.random.rand(240, 426, 3) for _ in range(80)] # List of ndarrays

    n_frames_to_load = 250
    fps = 25
    indices = list(range(0, n_frames_to_load * fps, fps))
    video_path = '/media/zeleznyt/DATA/repo/ViViT/example_data_RAVDAI/ct24 2023-10-02 01.27.34_example.mp4'
    vr = decord.VideoReader(video_path)
    input_video = [vr[i].asnumpy() for i in indices]

    for i, input_frame in enumerate(input_video):
        start_time = time.time()
        result, probs = predictor.predict_class_with_probs(input_frame)
        probs_rounded = {k: f"{v:.2f}" for k, v in probs.items()}
        print(f'Predicted class at {frame_to_timestamp(i-8)}: {result}. All probabilities: {probs_rounded}')
        print('Time elapsed during inference:', time.time() - start_time)