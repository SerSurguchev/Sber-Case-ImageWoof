%%writefile dataset.py
from PIL import Image
import torch
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms

"""
Dictionary to rename labels in column
"""
lbl_dict = dict(
    n02115641=0,
    n02086240=1,
    n02088364=2,
    n02087394=3,
    n02105641=4,
    n02111889=5,
    n02099601=6,
    n02096294=7,
    n02093754=8,
    n02089973=9
)


def create_train_val_csv(csv_file_dir):
    """
    Function which creates 2 csv files with train and valid images and labels

    Parameters
    :param csv_file_dir: (str): Path to csv file
    """

    df = pd.read_csv(csv_file_dir)

    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    df['noisy_labels_0'] = df['noisy_labels_0'].map(lbl_dict)

    df_train = df.loc[df.is_valid == False]
    df_val = df.loc[df.is_valid == True]

    df_train = df_train.iloc[:, :2].reset_index(drop=True)
    df_val = df_val.iloc[:, :2].reset_index(drop=True)

    # Create test dataset
    test_len = int(len(df_val) * 0.2)
    test_drop_indices = np.random.choice(df_val.index, test_len, replace=False)
    df_test = pd.DataFrame(df_val.loc[test_drop_indices])
    df_val = df.drop(test_drop_indices)

    df_train.to_csv('imagewoof_train.csv')
    df_val.to_csv('imagewoof_val.csv')
    df_test.to_csv('imagewoof_test.csv')


class CreateDataset(torch.utils.data.Dataset):
    """
    Class to create custom dataset
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')

        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            rgb_im = self.transform(rgb_im)

        return (rgb_im, y_label)


dataset_transform = {

    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[np.random.permutation(3), :, :]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),

    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
}