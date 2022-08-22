import torchvision
import torch.utils.data as data

import numpy as np
import requests
import argparse
import json
import time
import logging

from io import BytesIO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_dir', required=True, type=str,
                        help="ImageNet Validation Data Directory")
    parser.add_argument('--log_file', required=True, type=str,
                        help="Target Log File")
    parser.add_argument('--ip', required=True, type=str,
                        help="Target flask server ip")
    parser.add_argument('--port', required=True, type=str,
                        help="Target flask server port")
    parser.add_argument('--api', type=str, default='inference',
                        help="Target flask api")

    args = parser.parse_args()

    logger = logging.getLogger('FLASK')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(args.log_file, mode='a')
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    _format = 'PNG'
    s = time.time()
    infer_time_list = []
    resp_time_list = []
    dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')
    for i in range(10):
        img, _ = dataset[i]
        buffer = BytesIO()
        img.save(buffer, format=_format)
        img_bytes = buffer.getvalue()

        url = f'http://{args.ip}:{args.port}/{args.api}'
        files = {
            'image_bytes': img_bytes,
            # 'image_info': json.dumps({'mode': img.mode, 'size': img.size, 'format': _format})
        }
        ss = time.time()
        res = requests.post(url, files=files).json()
        resp_time = time.time() - s

        res['resp_time'] = resp_time
        infer_time_list.append(res['resp_time'])
        infer_time_list.append(res['infer_time'])

        logger.info(res)

    for hdlr in logger.handlers:
        hdlr.close()
