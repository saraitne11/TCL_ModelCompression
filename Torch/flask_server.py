import torch

from flask import Flask
from flask import request
from flask import jsonify

import argparse
import pickle
import json
from timeit import default_timer as timer


MODEL_DIR = './Models/'

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str,
                    help="PyTorch Model File Path (.pth file)")
parser.add_argument('--port', required=True, type=int,
                    help="Flask Server Port")
args = parser.parse_args()

model_file = args.model if args.model.endswith('.pth') else args.model + '.pth'
if 'script' in model_file:
    model = torch.jit.load(MODEL_DIR + model_file)
else:
    model = torch.load(MODEL_DIR + model_file)

with open('imagenet.json', 'r') as f:
    categories = json.loads(f.read())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

app = Flask(__name__)


@app.route('/model', methods=['GET'])
def get_model():
    return {'model': model_file}


@app.route('/inference', methods=['POST'])
def inference():
    files = request.files
    images_byte = files['images_byte'].read()
    x = pickle.loads(images_byte)       # [B, C, H, W]

    s = timer()
    x = x.to(device)
    output = model(x)
    infer_time = timer() - s

    _, top1 = output.max(1)
    top1_id = top1.tolist()
    # top1_name = list(map(lambda _id: categories[_id], top1_id))

    _, top5 = output.topk(5, 1, True, True)
    top5_id = top5.tolist()
    # top5_name = []
    # for b in top5_id:
    #     top5_name.append(list(map(lambda _id: categories[_id], b)))

    res = {
        'top1_id': top1_id,
        # 'top1_name': top1_name,
        'top5_id': top5_id,
        # 'top5_name': top5_name,
        'infer_time': infer_time
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, threaded=False)
