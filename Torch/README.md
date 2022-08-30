# Model Compression Evaluation - PyTorch

## R&R
### https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
TensorRT는 Container 내에서 작업 필수 (Local System 보호)
 - 박용서 M: TensorRT-Desktop
   - torch -> torch_trt (FP32, FP16, INT8)
     - densenet201-torch_trt_fp16.pth
   - torch -> onnx -> onnx_trt (FP32, FP16, INT8)
     - densenet201-onnx_trt_fp16.plan


- 윤정식 M: TensorRT-JetsonNano
  - torch -> torch_trt (FP32, FP16, INT8)
    - densenet201-jn-torch_trt_fp16.pth
  - torch -> onnx -> onnx_trt (FP32, FP16, INT8)
    - densenet201-jn-onnx_trt_fp16.plan


- 김경록 M: Pruning
  - densenet201-pruning.pth


- 이장원 M: Triton Client 개발
- 이장원 M: Desktop 실험
- Jetson Nano 실험


## Environments
- Python3.6 or later
- PyTorch 1.10.1
- torchvision 0.11.2

## Models
imagenet classification Models from `torchvision 0.11.2` 
- torchvision.models.resnet152
- torchvision.models.densenet201
- torchvision.models.efficientnet_b7
- torchvision.models.mobilenet_v2

## Data
ILSVRC2012 validation images (50,000 image) 

## Devices
- AMD 3990X ThreadRipper / Nvidia RTX 3090 24GB / RAM 128GB
- Nvidia Jetson Nano 4GB

## Model Servings
- Flask
- Nvidia Triton (TBD)

## Model Compression
- TensorRT
- Pruning
- Quantization

### Flask Serving
- Build Docker Image
```bash
$ cd TCL_ModelCompression/Torch
$ sudo docker build -t flask_app/torch .
```
- Run Docker Container
```bash
$ sudo docker run -d --gpus all -p <host port>:<container port> --name flask_app flask_app/torch python flask_server.py --model <.pth file> --port <container port>
# Example
$ sudo docker run -d --gpus all -p 2222:2222 --name flask_app flask_app/torch python flask_server.py --model resnet152-base.pth --port 2222
```
- Check flask_app response
```bash
$ curl -X GET "http://<ip>:<port>/model"
# Example
$ curl -X GET "http://127.0.0.1:2222/model"
# {"model":"resnet152-base.pth"}
```
- Check flask_app Process ID using `nvidia-smi`
- Run `process_monitor.py` for gpu memory usage of torch model
```bash
$ nohup python ../process_monitor.py --target_pid <pid> --log_file <log_file> &
# Example
$ nohup python ../process_monitor.py --target_pid 31888 --log_file monitors/resnet152-base.log &
```
- Run `flask_client.py`
```bash
$ python flask_client.py --imagenet_dir <dir> --log_file <log_file> --ip <flask_app_ip> --port <flask_app_port> --batch_size <batch_size>
# Example
$ python flask_client.py --imagenet_dir /home/data/ImageNet/ --log_file results/resnet152-base.log --ip localhost --port 2222 --batch_size 2
```

### Nvidia Triton Serving
```
TBD
```
