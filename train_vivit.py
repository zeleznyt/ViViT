import numpy as np
import torch
from torch import nn
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from vivit import ViViT
from dataset import VideoDataset
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from torch.utils.data import Subset
from utils import *

np.random.seed(42)

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def evaluate(model, data_loader, loss_func, device):
    model.eval()
    loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (data, target, padding_mask) in enumerate(data_loader):
            # Use this to visualize th data
            # visualize_frames(data.numpy()[0], CLASSES[target[0].numpy()])
            optimizer.zero_grad()
            x = data.to(device)
            padding_mask = padding_mask.to(device)
            data = rearrange(x, 'b p h w c -> b p c h w')
            target = target.type(torch.LongTensor).to(device)

            pred = model(data.float(), padding_mask)
            # logits = np.squeeze(pred.cpu().detach().numpy())

            loss += loss_func(pred, target).item()

            predicted_class = pred.argmax(dim=1)  # Get the predicted class
            correct_predictions += (predicted_class == target).sum().item()  # Count correct predictions
            total_predictions += target.size(0)  # Total number of predictions

        # print(target.item(), pred.argmax().item())
        # print(logits)
        loss = loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
    return loss, accuracy


def train_epoch(epoch, model, optimizer, train_data_loader, eval_data_loader, loss_history, loss_func, device, checkpoint_save_dir, log_step=100, eval_step=-1, save_step=-1, report_to=None):
    total_samples = len(train_data_loader.dataset)
    model.train()

    start_time = time.time()
    for i, (data, target, padding_mask) in enumerate(train_data_loader):
        # Use this to visualize th data
        # visualize_frames(data.numpy()[0], CLASSES[target[0].numpy()])
        optimizer.zero_grad()
        x = data.to(device)
        padding_mask = padding_mask.to(device)
        data = rearrange(x, 'b p h w c -> b p c h w')
        target = target.type(torch.LongTensor).to(device)

        pred = model(data.float(), padding_mask)

        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        lr_sched.step()

        end_time = time.time()

        if i % log_step == 0 or i == len(train_data_loader) - 1:
            # Log to wandb
            if report_to == 'wandb':
                wandb.log({"train_loss": loss.item(),
                           "time_per_iteration": (end_time - start_time) / log_step,
                           "epoch": epoch,
                           "learning_rate": optimizer.param_groups[0]['lr']})

            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(train_data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            start_time = time.time()

        if (i % eval_step == 0 or i == len(train_data_loader) - 1) and eval_step != -1:
            print('Evaluation started.')
            eval_loss, acc = evaluate(model, eval_data_loader, loss_func, device)
            print(f'Eval loss: {eval_loss:.4f}, eval accuracy: {acc:.4f}')

        if (i % save_step == 0 or i == len(train_data_loader) - 1) and save_step != -1:
            model_path = os.path.join(checkpoint_save_dir, 'model_{}-{}.pt'.format(epoch, i))
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print('Model successfully saved to {}'.format(model_path))


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


def create_balanced_subset(dataset, n_of_instances=-1):
    class_indices = defaultdict(list)
    for idx, data in enumerate(dataset):
        _, class_label, _ = data
        class_indices[class_label].append(idx)
    min_class_count = min(len(indices) for indices in class_indices.values())
    if n_of_instances < 0 or n_of_instances > min_class_count:
        n_of_instances = min_class_count
    balanced_indices = []
    for class_label, indices in class_indices.items():
        balanced_indices.extend(random.sample(indices, n_of_instances))
    balanced_subset = Subset(dataset, balanced_indices)
    print('Balanced subset created with {} instances for each class.'.format(n_of_instances))
    return balanced_subset


if __name__ == "__main__":
    # Process args and config
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    train_config = config['training']

    num_classes = len(CLASSES)
    model_config['num_classes'] = num_classes
    num_epochs = train_config['epochs']
    warmup_epochs = train_config['warmup_epochs']
    learning_rate = train_config['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViViT(model_config).to(device)
    # Move non-trainable mask to the device
    model.temporal_transformer.cls_mask = model.temporal_transformer.cls_mask.to(device)

    # Create dataset
    print('Loading dataset...')
    dataset = VideoDataset(data_config['meta_file'], CLASSES, frame_sample_rate=data_config['frame_sample_rate'],
                           min_sequence_length=data_config['min_sequence_length'],
                           max_sequence_length=data_config['max_sequence_length'],
                           video_decoder=data_config['video_decoder'],)
    if train_config['balance_dataset']:
        dataset = create_balanced_subset(dataset)
    train_dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=data_config['shuffle'],
                                  drop_last=data_config['drop_last'], num_workers=data_config['num_workers'])
    print('Dataset successfully loaded.')

    # dataset_distribution(dataset, True)

    # Set Loss, optimizer and cheduler
    criterion = nn.CrossEntropyLoss()
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

    train_loss_history, test_loss_history = [], []

    model_name = 'ViVit-B_{}x{}'.format(model_config['patch_size'], model_config['tubelet_size'])

    project_name = 'ViViT'
    if train_config['report_to'] == 'wandb':
        init_wandb(project_name, config)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        train_epoch(epoch, model, optimizer,
                    train_data_loader=train_dataloader,
                    eval_data_loader=train_dataloader,
                    loss_history=train_loss_history,
                    loss_func=criterion,
                    device=device,
                    log_step=train_config['log_step'], eval_step=train_config['eval_step'],
                    save_step=train_config['save_step'],
                    checkpoint_save_dir=os.path.join(train_config['checkpoint_save_dir'], model_name),
                    report_to=train_config['report_to'])

    print('Training finished.')
    model_path = os.path.join(os.path.join(train_config['checkpoint_save_dir'], model_name), 'model_final.pt')
    os.makedirs(os.path.join(train_config['checkpoint_save_dir'], model_name), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print('Model successfully saved to {}'.format(model_path))

