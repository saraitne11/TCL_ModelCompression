# flask_server.py

import torch
import torchvision
from flask import Flask
from flask import request
import json

import argparse


model_list = ['resnet', 'efficientnet', 'densenet', 'mobilenet']
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str,
                    help="One of [resnet, efficientnet, densenet, mobilenet]")
args = parser.parse_args()

if args.model == 'resnet':
    w = torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet152(weights=w)
elif args.model == 'efficientnet':
    w = torchvision.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_l(weights=w)
elif args.model == 'densenet':
    w = torchvision.models.densenet.DenseNet201_Weights.IMAGENET1K_V1
    model = torchvision.models.densenet201(weights=w)
elif args.model == 'mobilenet':
    w = torchvision.models.mobilenetv2.MobileNet_V2_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v2(weights=w)
else:
    raise ValueError(f"--model parameter must be in {model_list}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    files = request.files
    img_bytes = files['image_bytes']
    print(img_bytes.read())
    img_info = files['image_info']
    print(img_info)
    # _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
    # return str(result.item())
    return 'response'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222, threaded=False)
