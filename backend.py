import io
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return 'Welcome to prediction'

if __name__ == '__main__':
    app.run(debug=True)
