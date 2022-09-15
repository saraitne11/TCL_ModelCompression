import torch
import torchvision
import torch.utils.data as data

import numpy as np
import pickle
import os
import requests
import argparse
import logging
from timeit import default_timer as timer

from io import BytesIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-dir', required=True, type=str,
                        help="ImageNet Validation Data Directory")
    parser.add_argument('--log-file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--ip', required=True, type=str,
                        help="Target flask server ip")
    parser.add_argument('--port', required=True, type=str,
                        help="Target flask server port")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch Size")
    parser.add_argument('--transform-dir', type=str, default='./Transforms/',
                        help="Transform file directory")
    parser.add_argument('--loader-workers', type=int, default=0,
                        help="DataLoader Workers")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    logger = logging.getLogger('FLASK_CLIENT')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(args.log_file, mode='a')
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    resp = requests.get(url=f'http://{args.ip}:{args.port}/model')
    model_file = resp.json()['model']
    transform = torch.load(os.path.join(args.transform_dir, model_file))

    s = timer()
    infer_time_list = []
    resp_time_list = []
    n_top1 = 0
    n_top5 = 0
    dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, transform=transform, split='val')
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers)
    n_data = len(dataset)
    cnt_data = 0
    cnt_request = 0
    for images, labels in loader:
        images_byte = BytesIO(pickle.dumps(images))

        ss = timer()
        resp = requests.post(url=f'http://{args.ip}:{args.port}/inference',
                             files={'images_byte': images_byte})
        resp_time = timer() - ss

        res = resp.json()
        res['resp_time'] = resp_time
        infer_time_list.append(res['infer_time'])
        resp_time_list.append(res['resp_time'])

        n_top1 += np.equal(res['top1_id'], labels).sum()
        n_top5 += np.max(np.isin(res['top5_id'], labels), 1).sum()

        cnt_data += len(labels)
        cnt_request += 1
        logger.info(f"BatchSize: {len(labels)}, Progress: {cnt_data}/{n_data}, "
                    f"InferTime: {res['infer_time']:0.4f}, RespTime: {res['resp_time']:0.4f}")
        # logger.info(f"BatchSize: {len(labels)}, Progress: {i}/{n_data}, "
        #             f"InferTime: {res['infer_time']:0.4f}, RespTime: {res['resp_time']:0.4f}, "
        #             f"Top1: {res['top1_id']}, Top5: {res['top5_id']}")

    total_time = timer() - s
    total_infer_time = sum(infer_time_list)
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


if __name__ == '__main__':
    main()
