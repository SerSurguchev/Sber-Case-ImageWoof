% % writefile
utils.py
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import random
import config
import numpy as np


def save_plots(train_acc, valid_acc, train_loss, valid_loss, net_list):
    """
    Function to save the loss and accuracy plots to disk
    Parameters:
    :param train_acc: Python dict containing accuracy on training
    :param valid_acc: Python dict containing accuracy on validation
    :param train_loss: Python dict containing loss value on training
    :param valid_loss: Python dict containing loss value on validation
    :return:
    """
    # Accuracy plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    for experiment_id in net_list:
        axes[0][0].plot(train_acc[experiment_id], label=experiment_id)

    axes[0][0].legend()
    axes[0][0].set_title('Training accuracy')
    fig.tight_layout()

    for experiment_id in net_list:
        axes[0][1].plot(train_loss[experiment_id], label=experiment_id)

    axes[0][1].legend()
    axes[0][1].set_title('Training loss')
    fig.tight_layout()

    for experiment_id in net_list:
        axes[1][0].plot(valid_acc[experiment_id], label=experiment_id)

    axes[1][0].legend()
    axes[1][0].set_title('Validation accuracy')
    fig.tight_layout()

    for experiment_id in net_list:
        axes[1][1].plot(valid_loss[experiment_id], label=experiment_id)

    axes[1][1].legend()
    axes[1][1].set_title('Validation loss')
    fig.tight_layout()


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_val_loss,
                 model, optimizer, criterion, file):
        if current_val_loss < self.best_valid_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }, file)


def save_model(model, optimizer, criterion,
               file=config.FINAL_MODEL):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, file)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)