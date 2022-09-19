import torch
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

import argparse
import pickle
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('--model-repository', type=str,
                    help="Model Repository(Dir)")
parser.add_argument('--model', required=True, type=str,
                    help="Model File Name")
parser.add_argument('--port', required=True, type=int,
                    help="Flask Server Port")
args = parser.parse_args()


model_repository = args.model_repository if args.model_repository.endswith('/') else args.model_repository + '/'

if '-script' in args.model:
    model_file = args.model if args.model.endswith('.pt') else args.model + '.pt'
    model = torch.jit.load(model_repository + model_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    def infer(x):
        x = x.to(device)
        return model(x)

    def get_top1_id(output):
        _, top1 = output.max(1)
        top1_id = top1.tolist()
        return top1_id

    def get_top5_id(output):
        _, top5 = output.topk(5, 1, True, True)
        top5_id = top5.tolist()
        return top5_id

elif '-trt' in args.model:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt

    model_file = args.model if args.model.endswith('.plan') else args.model + '.plan'

    TRT_LOGGER = trt.Logger()
    trt_runtime = trt.Runtime(TRT_LOGGER)
    with open(model_repository + model_file, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    input_shape, input_dtype = engine.get_binding_shape(0), engine.get_binding_dtype(0)
    input_hmem = cuda.pagelocked_empty(trt.volume(input_shape), trt.nptype(input_dtype))
    input_dmem = cuda.mem_alloc(input_hmem.nbytes)

    output_shape, output_dtype = engine.get_binding_shape(1), engine.get_binding_dtype(1)
    output_hmem = cuda.pagelocked_empty(trt.volume(output_shape), trt.nptype(output_dtype))
    output_dmem = cuda.mem_alloc(output_hmem.nbytes)

    bindings = [int(input_dmem), int(output_dmem)]
    context = engine.create_execution_context()
    stream = cuda.Stream()

    def infer(x):
        np.copyto(input_hmem, x.ravel())
        cuda.memcpy_htod_async(input_dmem, input_hmem, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_hmem, output_dmem, stream)
        stream.synchronize()
        return output_hmem.reshape(output_shape)

    def get_top1_id(output):
        top1_id = output.argmax(1).tolist()
        return top1_id

    def get_top5_id(output):
        top5_id = output.argsort(1)[:, ::-1][:, :5].tolist()
        return top5_id

else:
    raise ValueError(f'Not Supported Model File: {args.model}')


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
    output = infer(x)
    infer_time = timer() - s

    top1_id = get_top1_id(output)
    top5_id = get_top5_id(output)

    res = {
        'top1_id': top1_id,
        'top5_id': top5_id,
        'infer_time': infer_time
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, threaded=False)
