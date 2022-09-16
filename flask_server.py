import torch

from flask import Flask
from flask import request
from flask import jsonify

import argparse
import pickle
import json
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('--model-repository', type=str,
                    help="Model Repository(Dir)")
parser.add_argument('--model', required=True, type=str,
                    help="Model File Name")
parser.add_argument('--port', required=True, type=int,
                    help="Flask Server Port")
args = parser.parse_args()


model_file = args.model if args.model.endswith('.pt') else args.model + '.pt'
model_repository = args.model_repository if args.model_repository.endswith('/') else args.model_repository + '/'
if 'script' in model_file:
    model = torch.jit.load(model_repository + model_file)
else:
    model = torch.load(model_repository + model_file)

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

    _, top5 = output.topk(5, 1, True, True)
    top5_id = top5.tolist()

    res = {
        'top1_id': top1_id,
        'top5_id': top5_id,
        'infer_time': infer_time
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, threaded=False)
