import torch

CSV_FILE_DIR = '/kaggle/working/noisy_imagewoof.csv'
ROOT_DIR = '/kaggle/working/imagewoof2/'

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
LEARNING_RATE = 1e-3
EPOCHS = 30
BEST_MODEL = 'model.pth'
FINAL_MODEL = 'final.pth'

PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 2,
          'pin_memory': True}