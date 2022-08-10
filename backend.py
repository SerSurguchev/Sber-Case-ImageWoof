from flask import Flask, request, render_template
from dataset import CreateDataset, dataset_transform
import config
import torch
from tqdm import tqdm
import io
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader
)

from resnet_models import ResNet18


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_predictions(model, checkpoint, image):
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print('Testing...')

    tensor = transform_image(image)

    with torch.no_grad():
        outputs = model(tensor)

        _, preds = torch.max(outputs.data, 1)

    return preds


app = Flask(__name__)


@app.route('/')
def about():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    file = request.files['file']
    img_bytes = file.read()

    pred = get_predictions(
        model=ResNet18(pretrained=False,
                       fine_tune=False,
                       num_classes=10),
        checkpoint='Resnet18_best.pth',
        image=img_bytes
    )

    return render_template('prediction.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)
