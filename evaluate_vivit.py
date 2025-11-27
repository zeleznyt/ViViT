import torch
from torch import nn
from torch.utils.data import DataLoader
from vivit import ViViT
from dataset import VideoDataset, VideoStreamDataset
from train_vivit import evaluate, plot_confusion_matrix
from utils.train_utils import *
from dataset import get_eaf, get_label_on_idx, get_video_length_opencv, preprocess_video
from train_vivit import compute_per_class_metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import json
import decord
from einops import rearrange

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def evaluate_from_dataset():
    print('Loading dataset...')
    assert data_config['dataset_type'] in ['one_class', 'stream'], f'Dataset type {data_config["dataset_type"]} not supported'
    if data_config['dataset_type'] == 'one_class':
        test_dataset = VideoDataset(data_config['test_meta_file'], CLASSES,
                            load_from_json = data_config['test_json'],
                            frame_sample_rate = data_config['frame_sample_rate'],
                            min_sequence_length = data_config['min_sequence_length'],
                            max_sequence_length = data_config['max_sequence_length'],
                            num_threads = data_config['decord_num_threads'],
                            normalize = data_config['normalize'],)
    elif data_config['dataset_type'] == 'stream':
        test_dataset = VideoStreamDataset(data_config['test_meta_file'], CLASSES,
                            load_from_json = data_config['test_json'],
                            frame_sample_rate = data_config['frame_sample_rate'],
                            context_size = data_config['context_size'],
                            overlap = data_config['overlap'],
                            max_empty_frames = data_config['max_empty_frames'],
                            num_threads = data_config['decord_num_threads'],
                            normalize = data_config['normalize'],)
    test_dataloader = DataLoader(test_dataset, batch_size=data_config['batch_size'], shuffle=False,
                                drop_last=False, num_workers=data_config['num_workers'])
    print('Dataset "{}" successfully loaded.'.format(data_config['dataset_type']))

    loss_func = nn.CrossEntropyLoss()

    print('Evaluation started.')
    eval_loss, acc, confusion, precision, recall, f1, per_class_metrics, per_class_accuracy = evaluate(model, test_dataloader, loss_func, device)

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
    return score


def evaluate_raw_video(video_path, annotation_path, input_fps=25, frame_sample_rate=1, context_size=8, overlap=16, step=1000):
    annotation_list = get_eaf(annotation_path)

    sampling = input_fps * frame_sample_rate

    mid_frame = context_size * sampling
    video_len = get_video_length_opencv(video_path)
    max_data_len = min(annotation_list[-1][1], video_len / input_fps * step)

    # decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    frames = []
    for idx in range(mid_frame - (context_size * sampling), mid_frame + ((context_size + 1) * sampling), sampling):
        frames.append(vr[idx].asnumpy())

    padding_mask = torch.tensor([False] * 17).unsqueeze(0)
    idx = mid_frame + context_size * sampling

    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []

    from tqdm import tqdm

    # Initialize progress bar
    total_steps = (max_data_len // step * input_fps - idx) // sampling - (len(frames) - 2)
    pbar = tqdm(total=total_steps, desc="Processing video")

    while idx / input_fps * step < max_data_len:
        label = get_label_on_idx((idx-(context_size*input_fps))/input_fps*step, annotation_list)
        if label == -1 or label == 'nedefinováno':
            idx += input_fps*frame_sample_rate
            continue
        label = torch.tensor(CLASSES.index(label))

        processed_video = preprocess_video(np.stack(frames, axis=0), normalize=True)
        processed_video = rearrange(np.stack(processed_video), 't h w c -> t c h w')
        processed_video = torch.from_numpy(processed_video).float()
        processed_video = processed_video.unsqueeze(0)

        data, target, padding_mask = [t.to(device) for t in (processed_video, label.unsqueeze(0), padding_mask)]

        pred = model(data.float(), padding_mask)

        predicted_class = pred.argmax(dim=1)  # Get the predicted class
        all_predictions.extend(predicted_class.cpu().numpy())  # Save predictions
        all_targets.extend(target.cpu().numpy())  # Save ground truth

        # Count correct predictions and Total number of predictions
        correct_predictions += (predicted_class == target).sum().item()
        total_predictions += target.size(0)

        frames.append(vr[idx + sampling].asnumpy())
        # Drop the first one
        frames.pop(0)

        idx += sampling
        # Update progress bar
        pbar.update(1)

    pbar.close()

    accuracy = correct_predictions / total_predictions

    confusion, per_class_metrics, per_class_accuracy = compute_per_class_metrics(all_targets, all_predictions, CLASSES)

    # Compute Precision, Recall, and F1 Score
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    print(f'Eval accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
    print("Per-class metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}, Accuracy={per_class_accuracy[class_name]}")

    model_name = eval_config['checkpoint'].split('/')[-2]
    save_dir = os.path.join(eval_config['output_path'], model_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.verbose:
        plot_path = os.path.join(save_dir, f'confusion_eval_raw_video.jpg')
        plot_confusion_matrix(confusion, CLASSES, plot_path)

    score = {'accuracy': accuracy,
             'precision': precision,
             'recall': recall,
             'f1': f1,
             'per_class_metrics': per_class_metrics,
             'per_class_accuracy': per_class_accuracy}
    with open(os.path.join(save_dir, f'score_raw_video.json'), 'w') as f:
        json.dump(score, f, indent=2)

    print("Confusion Matrix:\n", confusion)
    return score


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
    # device = "cpu"
    model = ViViT(model_config).to(device)
    # # Move non-trainable mask to the device
    # model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    checkpoint = torch.load(eval_config['checkpoint'], weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # score = evaluate_from_dataset()
    model.eval()

    video_list = []
    with open(data_config['test_meta_file'], 'r') as f:
        meta_data = json.load(f)
    for i, video in enumerate(meta_data):
        video_path = video['video']
        annotation_path = video['annotation']
        score = evaluate_raw_video(video_path, annotation_path)
        break