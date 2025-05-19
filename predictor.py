from vivit import ViViT
from dataset import preprocess_video
import torch
from einops import rearrange
import numpy as np
import time
from utils.train_utils import *


CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViViTpredictor():
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = get_device()
        self.classes = CLASSES

        num_classes = len(CLASSES)
        model_config['num_classes'] = num_classes

        print('Loading model...')
        self.model = ViViT(model_config).to(self.device)
        # Move non-trainable mask to the device
        self.model.temporal_transformer.cls_mask = self.model.temporal_transformer.cls_mask.to(self.device)

        # Load the model
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded on {self.device} device.')

    def predict(self, video: list) -> int:
        """
        Predicts a class of the middle frame of the video.
        :param video: List of video frames (ndarrays):
                        Expected length: 17
                        Frame shape: height x width x channels (standard decord output)
        :return: predicted_class (int)
        """
        processed_video = preprocess_video(video, normalize=True)
        processed_video = rearrange(np.stack(processed_video), 't h w c -> t c h w')
        processed_video = torch.from_numpy(processed_video).float().to(self.device)
        processed_video = processed_video.unsqueeze(0)

        prediction = self.model(processed_video,
                           padding_mask=torch.tensor([False] * processed_video.shape[1]).unsqueeze(0).to(self.device))
        predicted_class = prediction.argmax(dim=1)
        return predicted_class

    def class2string(self, cls):
        return self.classes[cls]

    def predict_class(self, video):
        cls = self.predict(video)
        predicted_class = self.class2string(cls)
        return predicted_class


if __name__ == '__main__':
    # Process args and config
    args = parse_args()
    config = load_config(args.config)
    model_config = config['model']
    checkpoint_path = config['evaluation']['checkpoint']
    predictor = ViViTpredictor(model_config, checkpoint_path)
    input_video = [np.random.rand(240, 426, 3) for _ in range(17)] # List of ndarrays
    start_time = time.time()
    result = predictor.predict_class(input_video)
    print('Predicted class: ', result)
    print('Time elapsed during inference:', time.time() - start_time)