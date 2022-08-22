import torchvision
import requests
import argparse
import json

from io import BytesIO


FORMAT = 'PNG'

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet_dir', required=True, type=str,
                    help="ImageNet Validation Data Directory")
parser.add_argument('--ip', type=str, default='127.0.0.1',
                    help="Target flask server ip")
parser.add_argument('--port', type=str, default='2222',
                    help="Target flask server port")
parser.add_argument('--api', type=str, default='inference',
                    help="Target flask api")
args = parser.parse_args()


dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')

img, _ = dataset[0]
buffer = BytesIO()
img.save(buffer, format=FORMAT)
img_bytes = buffer.getvalue()

files = {
    'image_bytes': img_bytes,
    'image_info': json.dumps({'mode': img.mode, 'size': img.size, 'format': FORMAT})
}

url = f'http://{args.ip}:{args.port}/{args.api}'
resp = requests.post(url, files=files)
print(resp.json)
