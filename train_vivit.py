import numpy as np
import torch
import json
from torch import nn
from einops import rearrange
from vivit import ViViT
from dataset import VideoDataset, VideoStreamDataset, VideoStreamFromEmbeddings
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from torch.utils.data import Subset, Sampler, Dataset, DataLoader
from utils.train_utils import *
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def plot_confusion_matrix(confusion, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a grid of colors based on the confusion matrix
    cax = ax.matshow(confusion, cmap="Blues")
    fig.colorbar(cax)  # Add a color bar for reference

    # Set axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path is not None:
        # Save the plot as an image
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        # Display the plot
        plt.show()
    return save_path


def target_to_string(target):
    # Transform to list
    if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
        target = target.tolist()
    elif not isinstance(target, list):
        raise TypeError("Unsupported type. Target must be a tensor, numpy array, or list.")

    # Check if target contains valid indices
    if not all(isinstance(idx, int) and 0 <= idx < len(CLASSES) for idx in target):
        raise ValueError("All elements in the target must be valid indices in the CLASSES array.")

    return [CLASSES[idx] for idx in target]


def compute_per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        correct = confusion[i, i]
        total = confusion[i].sum()
        acc = correct / total if total > 0 else 0.0
        per_class_accuracy[class_name] = round(acc, 4)

    # Per-class precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class_metrics = {
        class_names[i]: {
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4)
        }
        for i in range(len(class_names))
    }

    return confusion, per_class_metrics, per_class_accuracy


def evaluate(model, data_loader, loss_func, device, embeddings=False):
    model.eval()
    loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, (data, target, padding_mask) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Use this to visualize th data
            # visualize_frames(data.numpy()[0], CLASSES[target[0].numpy()])

            # Preprocess data and target
            data, target, padding_mask = [t.to(device) for t in (data, target, padding_mask)]

            # Model predictions
            if embeddings:
                pred = model(data, padding_mask)
            else:
                data = rearrange(data, 'b p h w c -> b p c h w')
                pred = model(data.float(), padding_mask)

            # Compute loss
            loss += loss_func(pred, target).item()

            # Collect predictions and ground truth
            predicted_class = pred.argmax(dim=1)  # Get the predicted class
            all_predictions.extend(predicted_class.cpu().numpy())  # Save predictions
            all_targets.extend(target.cpu().numpy())  # Save ground truth

            # Count correct predictions and Total number of predictions
            correct_predictions += (predicted_class == target).sum().item()
            total_predictions += target.size(0)

        # Compute final loss and accuracy
        loss = loss / len(data_loader)
        accuracy = correct_predictions / total_predictions

        confusion, per_class_metrics, per_class_accuracy = compute_per_class_metrics(all_targets, all_predictions, CLASSES)

        # Compute Precision, Recall, and F1 Score
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    return loss, accuracy, confusion, precision, recall, f1, per_class_metrics, per_class_accuracy


def log_metrics(_step, _metrics, _time_per_iteration, _epoch=None, _report_to='wandb', _train=True, _lr=None,
                _per_class_metrics=None, _per_class_accuracy=None):
    """
    Handles all logging to wandb and printing.
    """
    if _train:
        if _report_to == 'wandb':
            wandb.log({
                "train/loss": _metrics['loss'],
                "train/time_per_iteration": _time_per_iteration,
                "train/epoch": _epoch,
                "train/learning_rate": _lr
            }, step=_step, commit=False)
    else:  # evaluation
        if _report_to == 'wandb':
            # Log per-class metrics
            if _per_class_metrics:
                print("Per-class metrics:")
                for c, m in _per_class_metrics.items():
                    wandb.log({
                        f"eval/per_class/{c}/precision": m['precision'],
                        f"eval/per_class/{c}/recall": m['recall'],
                        f"eval/per_class/{c}/f1": m['f1'],
                        f"eval/per_class/{c}/accuracy": _per_class_accuracy[c],
                    }, step=_step, commit=False)
                    print(f"{c}: P={m['precision']:.4f}, "
                          f"R={m['recall']:.4f}, F1={m['f1']:.4f}, "
                          f"Acc={_per_class_accuracy[c]:.4f}")
            wandb.log({"eval/loss": _metrics['loss'],
                       "eval/accuracy": _metrics['accuracy'],
                       "eval/precision": _metrics['precision'],
                       "eval/recall": _metrics['recall'],
                       "eval/f1": _metrics['f1'],
                       "eval/time_per_evaluation": _time_per_iteration},
                      step=_step, commit=False)


def train_epoch(epoch, model, optimizer, lr_sched, gradient_accumulation_steps, train_data_loader, eval_data_loader,  loss_func, device,
                checkpoint_save_dir, global_step=0, log_step=100, eval_step=-1, save_step=-1, report_to=None, eval_metric='f1', embeddings=False):
    assert eval_metric in ['loss', 'accuracy', 'f1'], "Metric must be one of: 'loss', 'accuracy', 'f1'"
    total_samples = len(train_data_loader.dataset)
    model.train()

    with open(os.path.join(checkpoint_save_dir, 'checkpoints.json'), 'r') as f:
        all_checkpoints_json = json.load(f)
    best_checkpoints = all_checkpoints_json['3-best']  # List to store best checkpoint paths with their metric values
    all_checkpoints_json['metric'] = eval_metric
    metric_sign = 1 if eval_metric == 'loss' else -1  # Loss is minimized; others are maximized
    metric_value = 0 if metric_sign == -1 else 100  # Initiate metric value

    def update_best_checkpoints(checkpoint_path, metric_value):
        all_checkpoints_json['all'][os.path.basename(checkpoint_path)] = metric_value
        best_checkpoints.append((checkpoint_path, metric_value))
        best_checkpoints.sort(key=lambda x: x[1] * metric_sign)  # Sort by metric value
        all_checkpoints_json['best_checkpoint'] = {os.path.basename(best_checkpoints[0][0]): best_checkpoints[0][1]}

        if len(best_checkpoints) > 3:
            ckpt_to_be_removed = best_checkpoints.pop(3)[0]
            if ckpt_to_be_removed not in [c[0] for c in best_checkpoints]:
                if ckpt_to_be_removed != checkpoint_path and os.path.exists(ckpt_to_be_removed):  # Do not remove last checkpoint
                    os.remove(ckpt_to_be_removed)  # Keep only top 3 based on metric
        latest_checkpoint = all_checkpoints_json['latest_checkpoint']
        if latest_checkpoint is not None and latest_checkpoint not in [c[0] for c in best_checkpoints]:
            if latest_checkpoint != checkpoint_path and os.path.exists(latest_checkpoint):
                os.remove(latest_checkpoint)  # Remove second latest ckpt if it is not in best 3
        all_checkpoints_json['latest_checkpoint'] = checkpoint_path
        all_checkpoints_json['3-best'] = best_checkpoints
        with open(os.path.join(checkpoint_save_dir, 'checkpoints.json'), 'w') as f:
            json.dump(all_checkpoints_json, f, indent=2)

    start_time = time.time()
    for i, (data, target, padding_mask) in enumerate(train_data_loader):
        global_step += 1
        # Use this to visualize the data
        # visualize_frames(data.numpy()[0], CLASSES[target[0].numpy()])
        data, target, padding_mask = [t.to(device) for t in (data, target, padding_mask)]
        if i % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        if embeddings:
            pred = model(data, padding_mask)
        else:
            data = rearrange(data, 'b p h w c -> b p c h w')
            pred = model(data.float(), padding_mask)

        loss = loss_func(pred, target)
        loss = loss / gradient_accumulation_steps  # Normalize loss
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_data_loader):
            optimizer.step()
            lr_sched.step()

        end_time = time.time()

        if global_step % eval_step == 0 and eval_step != -1:
            print('Evaluation started.')
            eval_start_time = time.time()
            eval_loss, acc, confusion, precision, recall, f1, per_class_metrics, per_class_accuracy = evaluate(model, eval_data_loader, loss_func, device, embeddings=embeddings)
            eval_end_time = time.time()
            log_metrics(_step=global_step, _time_per_iteration=eval_end_time - eval_start_time,
                        _metrics={'loss': eval_loss,
                                  'accuracy': acc,
                                  'f1': f1,
                                  'precision': precision,
                                  'recall': recall},
                        _report_to=report_to, _train=False,
                        _per_class_metrics=per_class_metrics, _per_class_accuracy=per_class_accuracy)

            print(f'Eval loss: {eval_loss:.4f}, eval accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
            print("Per-class metrics:")
            for class_name, metrics in per_class_metrics.items():
                print(f"{class_name}: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}, Accuracy={per_class_accuracy[class_name]}")

            metric_value = eval_loss if eval_metric == 'loss' else acc if eval_metric == 'accuracy' else f1

            if args.verbose:
                confusion_matrix_path = os.path.join(checkpoint_save_dir, 'confusion')
                os.makedirs(confusion_matrix_path, exist_ok=True)
                plot_path = os.path.join(confusion_matrix_path, 'confusion_{}-{}.jpg'.format(epoch, i))
                plot_confusion_matrix(confusion, CLASSES, plot_path)

        if global_step % save_step == 0 and save_step != -1:
            checkpoint_path = os.path.join(checkpoint_save_dir, 'model_{}-{}.pth'.format(epoch, i))
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_sched.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, checkpoint_path)
            update_best_checkpoints(checkpoint_path, metric_value)
            print('Model successfully saved to {}'.format(checkpoint_path))

        if global_step % log_step == 0:
            # Log to wandb
            log_metrics(_step=global_step, _epoch=epoch, _metrics={'loss': loss.item()},
                        _time_per_iteration=(end_time - start_time) / log_step,
                        _report_to=train_config['report_to'], _train=True,
                        _lr=optimizer.param_groups[0]['lr'])
            wandb.log({}, step=global_step, commit=True)

            print('[Epoch {:2}] ['.format(epoch) + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(train_data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            start_time = time.time()

    print('End of epoch.')
    return global_step


def dataset_distribution(dataset, plot=False):
    class_counts = {}
    print('Counting classes...')
    for data in dataset:
        _, class_label, _ = data

        if class_label not in class_counts:
            class_counts[class_label] = 0
        class_counts[class_label] += 1

    print("Number of samples per class:")
    for class_label, count in class_counts.items():
        print(f"Class {CLASSES[class_label]}: {count} samples")

    if plot:
        class_labels = list(class_counts.keys())
        class_labels = [CLASSES[c] for c in class_labels]
        counts = list(class_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, counts)

        plt.xlabel('Class Labels')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return class_counts


def create_balanced_subset(dataset, balance_n_classes=0, n_of_instances=-1):
    """
    Randomly samples original dataset indices to create a smaller dataset with balanced class count
    :param dataset: torch.utils.data.Dataset to be split
    :param balance_n_classes: balances the dataset so that each class has maximum of instances equal to the class
        with nth most instances. This is good to balance once class with extremely high number of instances compared
        to the other classes. 0 by default - no such balance is used.
    :param n_of_instances: Only used when balance_n_classes == 0. Sets a maximum number of instances for each class.
        -1 by default - the values is set to a number of instances in the least represented class.
    :return: Subset. Balanced dataset.
    """
    class_indices = defaultdict(list)
    for idx, data in enumerate(dataset):
        _, class_label, _ = data
        class_indices[class_label].append(idx)
    if balance_n_classes <= 0:
        min_class_count = min(len(indices) for indices in class_indices.values())
        if n_of_instances < 0 or n_of_instances > min_class_count:
            n_of_instances = min_class_count
        balanced_indices = []
        for class_label, indices in class_indices.items():
            balanced_indices.extend(random.sample(indices, n_of_instances))
        balanced_subset = Subset(dataset, balanced_indices)
        print('Balanced subset created with {} instances for each class.'.format(n_of_instances))
    else:
        sorted_lengths = sorted((len(v) for v in class_indices.values()), reverse=True)
        nth_longest_length = sorted_lengths[balance_n_classes - 1]
        balanced_indices = []
        for class_label, indices in class_indices.items():
            if len(indices) <= nth_longest_length:
                balanced_indices.extend(indices)
            else:
                balanced_indices.extend(random.sample(indices, nth_longest_length))
        balanced_subset = Subset(dataset, balanced_indices)
        print('Balanced subset created with maximum of {} instances for each class.'.format(nth_longest_length))
    return balanced_subset


def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy random
    torch.manual_seed(seed_value)  # PyTorch (CPU & CUDA)
    torch.cuda.manual_seed(seed_value)  # GPU-specific seed
    torch.cuda.manual_seed_all(seed_value)  # Multi-GPU safe
    torch.backends.cudnn.deterministic = True


class GroupedVideoSampler(Sampler):
    def __init__(self, data, videos_per_group=4, shuffle=True):
        """
        Args:
            data: list of samples, each like [video_path, indices, label]
            videos_per_group: how many videos to load into cache at once
            shuffle: shuffle videos and samples each epoch
        """
        self.data = data
        self.videos_per_group = videos_per_group
        self.shuffle = shuffle

        # Group dataset indices by video
        self.video_to_indices = {}
        for idx, item in enumerate(data):
            video_path = item[0]
            if video_path not in self.video_to_indices:
                self.video_to_indices[video_path] = []
            self.video_to_indices[video_path].append(idx)

        self.videos = list(self.video_to_indices.keys())

    def __iter__(self):
        videos = self.videos[:]
        if self.shuffle:
            random.shuffle(videos)

        # Go through groups of N videos
        for i in range(0, len(videos), self.videos_per_group):
            group = videos[i:i+self.videos_per_group]

            group_indices = []
            for v in group:
                idxs = self.video_to_indices[v]
                if self.shuffle:
                    random.shuffle(idxs)
                group_indices.extend(idxs)

            # Optionally shuffle across videos inside the group
            if self.shuffle:
                random.shuffle(group_indices)

            for idx in group_indices:
                yield idx

    def __len__(self):
        return len(self.data)


def create_datasets(dataset_type: str, **kwargs):
    data_config = kwargs['data_config']
    classes = kwargs['classes']
    if dataset_type == 'one_class':
        train_dataset = VideoDataset(data_config['train_meta_file'], classes,
                               load_from_json=data_config['train_json'],
                               frame_sample_rate=data_config['frame_sample_rate'],
                               min_sequence_length=data_config['min_sequence_length'],
                               max_sequence_length=data_config['max_sequence_length'],
                               num_threads=data_config['decord_num_threads'],
                               normalize=data_config['normalize'],)
        val_dataset = VideoDataset(data_config['val_meta_file'], classes,
                               load_from_json=data_config['val_json'],
                               frame_sample_rate=data_config['frame_sample_rate'],
                               min_sequence_length=data_config['min_sequence_length'],
                               max_sequence_length=data_config['max_sequence_length'],
                               num_threads=data_config['decord_num_threads'],
                               normalize=data_config['normalize'],)
    elif dataset_type == 'stream':
        train_dataset = VideoStreamDataset(data_config['train_meta_file'], classes,
                                     load_from_json=data_config['train_json'],
                                     frame_sample_rate=data_config['frame_sample_rate'],
                                     context_size=data_config['context_size'],
                                     overlap=data_config['overlap'],
                                     max_empty_frames=data_config['max_empty_frames'],
                                     num_threads=data_config['decord_num_threads'],
                                     max_readers=data_config['max_train_readers'],
                                     normalize=data_config['normalize'],)
        val_dataset = VideoStreamDataset(data_config['val_meta_file'], classes,
                                     load_from_json=data_config['val_json'],
                                     frame_sample_rate=data_config['frame_sample_rate'],
                                     context_size=data_config['context_size'],
                                     overlap=data_config['overlap'],
                                     max_empty_frames=data_config['max_empty_frames'],
                                     num_threads=data_config['decord_num_threads'],
                                     max_readers=1,
                                     normalize=data_config['normalize'],)
    elif dataset_type == 'embeddings':
        train_dataset = VideoStreamFromEmbeddings(load_from_json=data_config['train_json'],
                                                  classes=classes,
                                                  embeddings_file=data_config['embeddings_file'])
        val_dataset = VideoStreamFromEmbeddings(load_from_json=data_config['val_json'],
                                                classes=classes,
                                                embeddings_file=data_config['embeddings_file'])

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']

    set_seed(train_config['seed'])

    if train_config['model_name']:
        model_name = train_config['model_name']
    else:
        model_name = 'ViViT'

    if model_config.get('use_pretrained_encoder', False) == 'vit':
        spatial_name = 'ViT'
    elif model_config.get('use_pretrained_encoder', False) == 'resnet':
        spatial_name = 'RN50'
    else:
        spatial_name = str(model_config['spatial_num_layers'])
    model_name = '{}_S-{}_T-{}_{}x{}_{}'.format(model_name, spatial_name, model_config['temporal_num_layers'], model_config['patch_size'], model_config['tubelet_size'],
                                           datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    print('Model name: {}'.format(model_name))

    project_name = 'ViViT'
    if train_config['report_to'] == 'wandb':
        init_wandb(project_name, config, name=model_name)

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    warmup_epochs = train_config['warmup_epochs']
    learning_rate = train_config['learning_rate']

    model = ViViT(model_config, use_only_embeddings=True if data_config['dataset_type'] == 'embeddings' else False)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    model = model.to(device)
    if model_config.get('freeze_spatial_encoder'):
        for param in model.spatial_transformer.parameters():
            param.requires_grad = False
        print("Spatial encoder frozen.")

    # Create dataset
    start = time.time()
    print('Loading dataset...')
    assert data_config['dataset_type'] in ['one_class', 'stream', 'embeddings'], f'Dataset type {data_config["dataset_type"]} not supported'
    train_dataset, val_dataset = create_datasets(data_config['dataset_type'], classes = CLASSES, data_config=data_config)

    end = time.time()
    print('Dataset "{}" successfully loaded in {} seconds.'.format(data_config['dataset_type'], end - start))
    if train_config['balance_dataset'] and not os.path.exists(data_config['train_json']):
        start = time.time()
        print('Balancing training dataset...')
        train_dataset = create_balanced_subset(train_dataset)
        end = time.time()
        print('Dataset "{}" successfully balanced in {} seconds.'.format(data_config['dataset_type'], end - start))
    print('Dataset length: {}'.format(len(train_dataset)))

    train_sampler = GroupedVideoSampler(data=train_dataset.data, videos_per_group=data_config['max_train_readers'], shuffle=data_config['shuffle'])
    train_dataloader = DataLoader(train_dataset, batch_size=data_config['batch_size'], sampler=train_sampler,
                                  drop_last=data_config['drop_last'], num_workers=data_config['num_workers'],
                                  prefetch_factor=data_config['prefetch_factor'])
    val_dataloader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False,
                                drop_last=data_config['drop_last'], num_workers=data_config['num_workers'],
                                prefetch_factor=data_config['prefetch_factor'])


    # Set Loss, optimizer and scheduler
    if train_config['loss'] in ['cross_entropy', 'crossentropy']:
        if 'label_smoothing' in train_config:
            label_smoothing = train_config['label_smoothing']
        else:
            label_smoothing = 0.0
        if data_config['class_weights']:
            cls_weights = get_class_weights(data_config, CLASSES)
            cls_weights = torch.tensor(cls_weights, dtype=torch.float32).to(device)
        else:
            cls_weights = None
        criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
    elif train_config['loss'] == 'seesaw':
        from utils.seesaw_loss import SeesawLossWithLogits
        class_counts = get_class_counts(data_config, CLASSES)
        class_counts = [class_counts.get(cls, 0) for cls in CLASSES]
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        raise ValueError('Loss {} not recognized.'.format(train_config['loss']))
    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print('Unknown optimizer. Must be one of ["adam", "sgd"]. Setting "adam" instead...')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = len(train_dataloader)
    if train_config['lr_scheduler'] == 'cosine':
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=(num_epochs - warmup_epochs) * steps_per_epoch)
    elif train_config['lr_scheduler'] == 'constant':
        lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif train_config['lr_scheduler'] == 'multistep':
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75])
    else:
        print('Unknown optimizer. Must be one of ["cosine", "constant", "multistep"]. Setting "constant" instead]...')
        lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer)

    if train_config['warmup_epochs'] > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                                    total_iters=int(warmup_epochs * steps_per_epoch))
        lr_sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_sched],
                                 milestones=[int(warmup_epochs * steps_per_epoch)])

    start_epoch = 0
    # Load pretrained model
    if os.path.exists(train_config['load_from_checkpoint']):
        checkpoint = torch.load(train_config['load_from_checkpoint'], weights_only=train_config['load_only_weights'])

        model.load_state_dict(checkpoint['model_state_dict'])
        if not train_config['load_only_weights']:
            if 'optimizer_state_dict' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint.keys():
                lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint.keys():
                start_epoch = checkpoint['epoch']
            if 'loss' in checkpoint.keys():
                loss = checkpoint['loss']

        print(f'Model successfully loaded from {train_config["load_from_checkpoint"]}.')

    global_step = 0

    checkpoint_save_dir = os.path.join(train_config['checkpoint_save_dir'], model_name)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    all_checkpoints_json = {'metric': None, 'best_checkpoint': {}, 'latest_checkpoint': None, '3-best': [], 'all': {}}
    with open(os.path.join(checkpoint_save_dir, 'checkpoints.json'), 'w') as f:
        json.dump(all_checkpoints_json, f)

    for e in range(num_epochs):
        epoch = start_epoch + e
        print('Epoch:', epoch)
        global_step = train_epoch(epoch, model, optimizer, lr_sched,
                                         gradient_accumulation_steps = train_config['gradient_accumulation_steps'],
                                         train_data_loader=train_dataloader,
                                         eval_data_loader=val_dataloader,
                                         global_step=global_step,
                                         loss_func=criterion,
                                         device=device,
                                         log_step=train_config['log_step'],
                                         eval_step=train_config['eval_step'],
                                         save_step=train_config['save_step'],
                                         checkpoint_save_dir=checkpoint_save_dir,
                                         report_to=train_config['report_to'],
                                         embeddings=True if data_config['dataset_type'] == 'embeddings' else False,)

    print('Training finished.')
    model_path = os.path.join(os.path.join(train_config['checkpoint_save_dir'], model_name), 'model_final.pt')
    os.makedirs(os.path.join(train_config['checkpoint_save_dir'], model_name), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print('Model successfully saved to {}'.format(model_path))
