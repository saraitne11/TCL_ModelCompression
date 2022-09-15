import torch
import torchvision
import torch.utils.data as data

from tritonclient import http
from tritonclient import grpc

import numpy as np
import os
import argparse
import logging
import requests
from timeit import default_timer as timer

from typing import Union, Tuple, List


def get_ready_model(triton_client: Union[http.InferenceServerClient, grpc.InferenceServerClient]) -> Tuple[str, str]:
    model = ''
    version = ''
    if isinstance(triton_client, http.InferenceServerClient):
        model_repo = triton_client.get_model_repository_index()
    else:
        model_repo = triton_client.get_model_repository_index(as_json=True)['models']

    for m in model_repo:
        if 'state' in m and m['state'] == 'READY':
            model = m['name']
            version = m['version']
            break
    if not model or not version:
        raise ValueError("Triton Server don't have Ready Model")
    return model, version


def get_model_io_info(triton_client: Union[http.InferenceServerClient, grpc.InferenceServerClient],
                      model: str, version: str, batch_size: str) -> Tuple[str, List[int], str, str]:
    if isinstance(triton_client, http.InferenceServerClient):
        model_meta = triton_client.get_model_metadata(model, version)
    else:
        model_meta = triton_client.get_model_metadata(model, version, as_json=True)

    input_name = model_meta['inputs'][0]['name']
    input_shape = model_meta['inputs'][0]['shape']
    input_shape[0] = batch_size
    input_shape = list(map(int, input_shape))
    input_dtype = model_meta['inputs'][0]['datatype']

    output_name = model_meta['outputs'][0]['name']
    return input_name, input_shape, input_dtype, output_name


def get_triton_metrics(ip: str, port: str) -> str:
    resp = requests.get(url=f'http://{ip}:{port}/metrics')
    metrics = resp.text
    resp.close()
    return metrics


def get_total_infer_time(ip: str, metrics_port: str, model: str) -> float:
    target = 'nv_inference_request_duration_us'
    metrics = get_triton_metrics(ip, metrics_port)
    lines = list(filter(lambda x: target in x and model in x, metrics.split('\n')))
    if len(lines) == 0:
        return 0.0
    _line = lines[0]
    infer_time = float(_line[_line.rindex(' ')+1:]) / 10**6
    return infer_time


def _idx(_s):
    return int(_s.decode().split(':')[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-dir', required=True, type=str,
                        help="ImageNet Validation Data Directory")
    parser.add_argument('--log-file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--ip', required=True, type=str,
                        help="Target triton server ip")
    parser.add_argument('--port', required=True, type=str,
                        help="Target triton server port")
    parser.add_argument('--protocol', required=True, type=str,
                        help="Triton Inference Protocol HTTP or gRPC")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch Size")
    parser.add_argument('--transform-dir', type=str, default='./Transforms/',
                        help="Transform file directory")
    parser.add_argument('--triton-metrics-port', type=str, default=8002,
                        help="Triton Metrics Port")
    parser.add_argument('--loader-workers', type=int, default=0,
                        help="DataLoader Workers")
    args = parser.parse_args()

    if args.protocol.lower() not in ['http', 'grpc']:
        raise ValueError("Argument --protocol must be one of ['HTTP', 'gRPC']")
    protocol = http if args.protocol.lower() == 'http' else grpc

    _dir = os.path.dirname(args.log_file)
    if _dir:
        os.makedirs(_dir, exist_ok=True)

    logger = logging.getLogger('TRITON_CLIENT')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(args.log_file, mode='a')
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    kwargs = {'url': f'{args.ip}:{args.port}'}
    if args.protocol.lower() == 'http':
        kwargs['connection_timeout'] = 300
        kwargs['network_timeout'] = 300
    client = protocol.InferenceServerClient(**kwargs)
    _model, _version = get_ready_model(client)

    transform = torch.load(os.path.join(args.transform_dir, _model + '.pt'))
    input_name, input_shape, input_dtype, output_name = get_model_io_info(client, _model, _version, args.batch_size)

    _input = protocol.InferInput(input_name, input_shape, input_dtype)
    _output = protocol.InferRequestedOutput(output_name, class_count=5)

    s = timer()
    resp_time_list = []
    n_top1 = 0
    n_top5 = 0
    dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, transform=transform, split='val')
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers)
    n_data = len(dataset)
    cnt_data = 0
    cnt_request = 0
    for images, labels in loader:
        # In Last Batch, batch size may be smaller due to remaining data
        if len(labels) != args.batch_size:
            input_shape[0] = len(labels)
            _input = protocol.InferInput(input_name, input_shape, input_dtype)

        _input.set_data_from_numpy(images.numpy())
        ss = timer()
        resp = client.infer(model_name=_model, model_version='1',
                            inputs=[_input], outputs=[_output])
        resp_time = timer() - ss
        resp_time_list.append(resp_time)

        res = list(map(lambda x: list(map(_idx, x)), resp.as_numpy(output_name)))
        top1_id = list(map(lambda x: x[0], res))
        top5_id = res

        n_top1 += np.equal(top1_id, labels).sum()
        n_top5 += np.max(np.isin(top5_id, labels), 1).sum()

        cnt_data += len(labels)
        cnt_request += 1
        logger.info(f"BatchSize: {len(labels)}, Progress: {cnt_data}/{n_data}, "
                    f"RespTime: {resp_time:0.4f}")

    total_time = timer() - s
    total_infer_time = get_total_infer_time(args.ip, args.triton_metrics_port, _model)
    total_resp_time = sum(resp_time_list)
    top1_acc = n_top1 / n_data
    top5_acc = n_top5 / n_data
    logger.info(f"TotalTime: {total_time:0.5f}")
    logger.info(f"TotalInferTime: {total_infer_time:0.5f}, "
                f"TotalRespTime: {total_resp_time:0.5f}")
    logger.info(f"AvgInferTime_ByData: {total_infer_time / cnt_data:0.5f}, "
                f"AvgRespTime_ByData: {total_resp_time / cnt_data:0.5f}")
    logger.info(f"AvgInferTime_ByRequest: {total_infer_time / cnt_request:0.5f}, "
                f"AvgRespTime_ByRequest: {total_resp_time / cnt_request:0.5f}")

    _sub = resp_time_list[:10]
    m, a = max(_sub), sum(_sub) / len(_sub)
    logger.info(f"First 10 [Max, Average] Response Time: [{m:0.5f}, {a:0.5f}]")
    _sub = resp_time_list[10:]
    m, a = max(_sub), sum(_sub) / len(_sub)
    logger.info(f"Remain [Max, Average] Response Time: [{m:0.5f}, {a:0.5f}]")

    logger.info(f"Top1Acc: {top1_acc:0.5f}, Top5Acc: {top5_acc:0.5f}")

    for hdlr in logger.handlers:
        hdlr.close()

    client.close()


if __name__ == '__main__':
    main()
