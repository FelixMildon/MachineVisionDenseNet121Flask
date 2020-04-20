import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
imagenet_class_index = json.load(open('D:/Programming/CVpytorch/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


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

def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("w00t1")
        if 'file' not in request.files:
            print("w00t2")
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            print("w00t3")
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        print("w00t4")
        print(file)
        print(img_bytes)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name, file=file)
    print("return index.html")
    return render_template('index.html')

if __name__ == '__main__':
    app.run()