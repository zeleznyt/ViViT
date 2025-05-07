import torch
from torch import nn
from torch.utils.data import DataLoader
from vivit import ViViT
from dataset import VideoDataset, VideoStreamDataset
from train_vivit import evaluate, plot_confusion_matrix
from utils.train_utils import *
import json

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    eval_config = config['evaluation']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)
    # Move non-trainable mask to the device
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    checkpoint = torch.load(eval_config['checkpoint'], weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading dataset...')
    assert data_config['dataset_type'] in ['one_class', 'stream'], f'Dataset type {data_config["dataset_type"]} not supported'
    if data_config['dataset_type'] == 'one_class':
        val_dataset = VideoDataset(data_config['test_meta_file'], CLASSES,
                               # load_from_json=data_config['val_json'],
                               # frame_sample_rate=data_config['frame_sample_rate'],
                               # min_sequence_length=data_config['min_sequence_length'],
                               # max_sequence_length=data_config['max_sequence_length'],
                               # video_decoder=data_config['video_decoder'],)
                            load_from_json = data_config['val_json'],
                            frame_sample_rate = data_config['frame_sample_rate'],
                            min_sequence_length = data_config['min_sequence_length'],
                            max_sequence_length = data_config['max_sequence_length'],
                            num_threads = data_config['decord_num_threads'],
                            normalize = data_config['normalize'],)
    elif data_config['dataset_type'] == 'stream':
        val_dataset = VideoStreamDataset(data_config['test_meta_file'], CLASSES,
                                     # load_from_json=data_config['val_json'],
                                     # frame_sample_rate=data_config['frame_sample_rate'],
                                     # context_size=data_config['context_size'],
                                     # overlap=data_config['context_size'],
                                     # max_empty_frames=data_config['max_empty_frames'],
                                     # video_decoder=data_config['video_decoder'],)
                            load_from_json = data_config['val_json'],
                            frame_sample_rate = data_config['frame_sample_rate'],
                            context_size = data_config['context_size'],
                            overlap = data_config['overlap'],
                            max_empty_frames = data_config['max_empty_frames'],
                            num_threads = data_config['decord_num_threads'],
                            normalize = data_config['normalize'],)
    val_dataloader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False,
                                drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])
    print('Dataset "{}" successfully loaded.'.format(data_config['dataset_type']))

    loss_func = nn.CrossEntropyLoss()

    print('Evaluation started.')
    eval_loss, acc, confusion, precision, recall, f1, per_class_metrics, per_class_accuracy = evaluate(model, val_dataloader, loss_func, device)

    print(f'Eval loss: {eval_loss:.4f}, eval accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
    print("Per-class metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}, Accuracy={per_class_accuracy[class_name]}")

    model_name = eval_config['checkpoint'].split('/')[-2]
    save_dir = os.path.join(eval_config['output_path'], model_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.verbose:
        plot_path = os.path.join(save_dir, f'confusion_eval.jpg')
        plot_confusion_matrix(confusion, CLASSES, plot_path)

    score = {'eval_loss': eval_loss,
             'accuracy': acc,
             'precision': precision,
             'recall': recall,
             'f1': f1,
             'per_class_metrics': per_class_metrics,
             'per_class_accuracy': per_class_accuracy}
    with open(os.path.join(save_dir, f'score.json'), 'w') as f:
        json.dump(score, f, indent=2)

    print("Confusion Matrix:\n", confusion)
