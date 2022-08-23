import torchvision
import torch.utils.data as data

import pickle
import os
import requests
import argparse
import logging
from timeit import default_timer as timer

from io import BytesIO

from torch_models import get_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_dir', required=True, type=str,
                        help="ImageNet Validation Data Directory")
    parser.add_argument('--log_file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--ip', required=True, type=str,
                        help="Target flask server ip")
    parser.add_argument('--port', required=True, type=str,
                        help="Target flask server port")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch Size")
    parser.add_argument('--loader_workers', type=int, default=2,
                        help="DataLoader Workers")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    logger = logging.getLogger('FLASK')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(args.log_file, mode='a')
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    resp = requests.get(url=f'http://{args.ip}:{args.port}/model')
    _model = resp.json()['model']
    transform = get_transform(_model)

    s = timer()
    infer_time_list = []
    resp_time_list = []
    n_top1 = 0
    n_top5 = 0
    dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, transform=transform, split='val')
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers)
    n_data = len(dataset)
    i = 0
    for images, labels in loader:
        images_byte = BytesIO(pickle.dumps(images))

        ss = timer()
        resp = requests.post(url=f'http://{args.ip}:{args.port}/inference',
                             files={'images_byte': images_byte})
        resp_time = timer() - ss

        res = resp.json()
        res['resp_time'] = resp_time
        infer_time_list.append(res['resp_time'])
        resp_time_list.append(res['infer_time'])

        n_top1 += 999
        n_top5 += 999

        i += len(labels)
        logger.info(f"BatchSize: {len(labels)}, Progress: {i}/{n_data}, "
                    f"InferTime: {res['infer_time']:0.4f}, RespTime: {res['resp_time']:0.4f}")
        # logger.info(f"BatchSize: {len(labels)}, Progress: {i}/{n_data}, "
        #             f"InferTime: {res['infer_time']:0.4f}, RespTime: {res['resp_time']:0.4f}, "
        #             f"Top1: {res['top1_id']}, Top5: {res['top5_id']}")

    total_time = timer() - s
    avg_infer_time = sum(infer_time_list)/len(infer_time_list)
    avg_resp_time = sum(resp_time_list)/len(resp_time_list)
    top1_acc = n_top1 / n_data
    top5_acc = n_top5 / n_data
    logger.info(f"TotalTime: {total_time:0.4f}, "
                f"AvgInferTime: {avg_infer_time:0.4f}, AvgRespTime: {avg_resp_time:0.4f}"
                f"Top1Acc: {top1_acc:0.4f}, Top5Acc: {top5_acc:0.4f}")

    for hdlr in logger.handlers:
        hdlr.close()


if __name__ == '__main__':
    main()
