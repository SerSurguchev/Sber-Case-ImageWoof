from dataset import CreateDataset, dataset_transform
import config
import torch
from tqdm import tqdm
import io
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template

from torch.utils.data import (
    Dataset,
    DataLoader
)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def test(model, image):
    """
    Function to test the model
    """
    # Set model to evaluation mode
    model.eval()
    print('Testing...')

    tensor = transform_image(image)
    with torch.no_grad():

            outputs = model(image)

            _, preds = torch.max(outputs.data, 1)

    return preds


def get_predictions(model_path):

    model_path.load_state_dict(checkpoint['model_state_dict'])


app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    return 'Welcome to prediction'


if __name__ == '__main__':
    app.run(debug=True)
