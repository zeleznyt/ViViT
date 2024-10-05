import torch
from torch import nn
from torch.utils.data import DataLoader
from vivit import ViViT
from dataset import VideoDataset
from train_vivit import evaluate
from utils.train_utils import *

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']
    eval_config = config['evaluation']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    warmup_epochs = train_config['warmup_epochs']
    learning_rate = train_config['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)
    # Move non-trainable mask to the device
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    model.load_state_dict(torch.load(eval_config['checkpoint']))
    print('Loading dataset...')
    val_dataset = VideoDataset(eval_config['dataset_meta_file'], CLASSES,
                               frame_sample_rate=data_config['frame_sample_rate'],
                               min_sequence_length=data_config['min_sequence_length'],
                               max_sequence_length=data_config['max_sequence_length'],
                               video_decoder=data_config['video_decoder'],)
    val_dataloader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
                                drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])
    print('Dataset successfully loaded.')

    loss_func = nn.CrossEntropyLoss()

    print('Evaluation started.')
    eval_loss, acc = evaluate(model, val_dataloader, loss_func, device)
    print(f'Eval loss: {eval_loss:.4f}, eval accuracy: {acc:.4f}')