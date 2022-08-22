import torch

from flask import Flask
from flask import request
from flask import jsonify

from PIL import Image
from io import BytesIO

import argparse
import json
import time

from torch_models import get_model


model_list = ['resnet', 'efficientnet', 'densenet', 'mobilenet']
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str,
                    help="One of [resnet, efficientnet, densenet, mobilenet]")
parser.add_argument('--port', required=True, type=int,
                    help="Flask Server Port")
args = parser.parse_args()

model, transform, categories = get_model(args.model)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    files = request.files
    image_bytes = files['image_bytes'].read()
    # image_info = json.loads(files['image_info'].read().decode('utf-8'))

    img = Image.open(BytesIO(image_bytes))
    x = transform(img).unsqueeze(0)     # [1, 3, 224, 224]
    x = x.to(device)
    s = time.time()
    output = model(x)
    infer_time = time.time() - s

    _, top1 = output.max(1)
    _, top5 = output.topk(5, 1, True, True)
    top1 = top1[0]      # 1 batch
    top5 = top5[0]      # 1 batch

    top1_id = top1.item()
    top1_name = categories[top1_id]

    top5_ids = top5.tolist()
    top5_names = list(map(lambda _id: categories[_id], top5_ids))

    res = {
        'top1_id': top1_id,
        'top1_name': top1_name,
        'top5_ids': top5_ids,
        'top5_names': top5_names,
        'infer_time': infer_time
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, threaded=False)
