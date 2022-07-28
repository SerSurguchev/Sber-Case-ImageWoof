
# Import from custom modules
import config
from utils import (
    save_plots,
    save_model,
    SaveBestModel,
    seed_everything,

)

from dataset import (
    CreateDataset,
    create_train_val_csv,
    dataset_transform
)

from models import (
    ResNet50,
    ResNet101,
    GoogLeNet,
    ResNet34,
    ResNet18
)

from train import train, validation

from torch.utils.data import (
    Dataset,
    DataLoader
)

# Import PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

seed_everything(42)


def main(model=ResNet50(img_channel=3, num_classes=10),
         file_name_to_save=config.BEST_MODEL,
         EPOCHS=config.EPOCHS
         ):

    create_train_val_csv(config.CSV_FILE_DIR)

    dataset_train = CreateDataset('imagewoof_train.csv', root_dir=config.ROOT_DIR,
                                  transform=dataset_transform['train'])

    dataset_val = CreateDataset('imagewoof_val.csv', root_dir=config.ROOT_DIR,
                                transform=dataset_transform['val'])

    train_loader = DataLoader(dataset_train, **config.PARAMS)

    valid_loader = DataLoader(dataset_val, **config.PARAMS)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    model = model
    model = model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    save_best_model = SaveBestModel()

    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {EPOCHS}")

        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)

        valid_epoch_loss, valid_epoch_acc = validation(model, valid_loader,
                                                       criterion)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        save_best_model(
            valid_epoch_loss, model, optimizer,
            criterion, file_name_to_save
        )

        print('-' * 15, ' Epoch complete ', '-' * 15)

    save_model(model, optimizer, criterion)

    print('TRAINING COMPLETE')

    return train_acc, valid_acc, train_loss, valid_loss
