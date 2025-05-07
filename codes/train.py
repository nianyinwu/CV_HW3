""" Training a Instance Segmentation model """

import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from utils import tqdm_bar
from eval import evaluation
from model import get_model
from dataloader import dataloader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ignore warnings
warnings.filterwarnings('ignore')

# Enable fast training
cudnn.benchmark = True

def get_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        '--device',
        type=str,
        choices=[
            "cuda",
            "cpu"
        ],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='../datas',
        help='the path of input data'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='the path of save the training model'
    )
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=30,
        help='number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=1e-4,
        help='learning rate'
    )

    return parser.parse_args()

def train(
    args: argparse.Namespace,
    cur_epoch: int,
    train_device: torch.device,
    train_model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    grad_scaler: torch.cuda.amp.GradScaler,
    scheduler: LRScheduler
) -> torch.Tensor:
    """
    Train the model for one epoch

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        cur_epoch (int): Current training epoch.
        train_device (torch.device): Device to train on (CPU or GPU).
        train_model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        grad_scaler (GradScaler):  Automatic mixed-precision (AMP) gradient scaler
        scheduler: Learning rate scheduler

    Returns:
        Tuple[torch.Tensor, float]: The average training loss and accuracy.
    """
    train_model.train()

    total_loss = 0

    for images, targets in (pbar := tqdm(data_loader, ncols=120)):

        images = [img.to(train_device) for img in images]
        targets = [{k: v.to(train_device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            loss_dict = train_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        grad_scaler.scale(losses).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        scheduler.step()


        total_loss += losses

        tqdm_bar('Train', pbar, losses.detach().cpu(), cur_epoch, args.epochs)


    avg_loss = total_loss / len(data_loader)

    return avg_loss

if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()


    # Ensure the save path exist
    os.makedirs(opt.save_path, exist_ok=True)

    model = get_model(num_classes=5).to(device)


    # Setting dataloader for training and validation
    train_loader = dataloader(args=opt, mode='train')
    val_loader = dataloader(args=opt, mode='valid')

    writer = SummaryWriter(log_dir=opt.save_path)

    # Setting the optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optim_func = optim.AdamW(
                    model.parameters(),
                    lr=opt.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=5e-4
                )

    total_steps = len(train_loader) * opt.epochs
    warmup_steps = int(total_steps * 0.25)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim_func,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_map = 0
    best_ap50 = 0

    val_gt_path = os.path.join(opt.data_path, 'gt.json')
    for epoch in range(opt.epochs):

        train_loss = train(
                        opt,
                        epoch,
                        device,
                        model,
                        train_loader,
                        optim_func,
                        scaler,
                        lr_scheduler
                    )

        mAP, AP50 = evaluation(device, model, val_loader, val_gt_path)


        current_lr = optim_func.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1}/{opt.epochs} | "
            f"Train Loss: {train_loss:.4f}  | "
            f"mAP: {mAP:.2%}    |"
            f"AP50: {AP50:.2%}  |"
            f"LR: {current_lr:.1e}"
        )

        if AP50 > best_ap50:
            best_ap50 = AP50
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'val_ap50_best.pth'))

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'val_mAP_best.pth'))


        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, f'epoch{epoch}.pth'))


        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Validation/mAP", mAP, epoch)
        writer.add_scalar("Validation/AP50", AP50, epoch)


    torch.save(model.state_dict(), os.path.join(opt.save_path, 'last.pth'))

    writer.close()
