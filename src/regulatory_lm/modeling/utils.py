import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, SequentialLR
from torch import nn
import math
import numpy as np
import random

def save_model(model, optimizer, validation_loss, epoch, save_path, scheduler=None):
    if scheduler is None:
        save_dict = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss
        }
    else:
        save_dict = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss
        }

    torch.save(save_dict, save_path)


def load_model(model, optimizer, load_path):
    load_dict = torch.load(load_path)
    if model is not None:
        model.load_state_dict(load_dict["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(load_dict["optimizer_state"])

    return model, optimizer, load_dict["epoch"], load_dict["validation_loss"]


def print_and_log(string, file):
    '''
    Handy function to print something to a terminal and write it to a log file
    '''
    print(string)
    file.write(string.replace(": ", ",").replace(" = ", ",") + "\n")


# def create_lr_scheduler(optimizer, total_steps, warmup_steps):
#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
#     return LambdaLR(optimizer, lr_lambda)

    
def create_lr_scheduler(optimizer, total_steps, warmup_steps, peak_lr, min_lr):
    print(total_steps, warmup_steps, min_lr)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

