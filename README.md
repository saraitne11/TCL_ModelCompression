# Model Compression Evaluation - PyTorch

# 참고 자료
- 220421 Nvidia AI Developer Meetup Hands-on
  - https://github.com/leejinho610/TRT_Triton_HandsOn
- Nvidia TensorRT Images
  - https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
- Torch-TensorRT
  - https://github.com/pytorch/TensorRT#compiling-torch-tensorrt
- Nvidia Triton Server
  - https://github.com/triton-inference-server/server/tree/main/docs

## R&R

TensorRT는 Container 내에서 작업 필수 (Local System 보호)
 - 박용서 M: TensorRT-Desktop
   - torch -> torch_trt (FP32, FP16, INT8)
     - densenet201-torch_trt_fp16.pth
   - torch -> onnx -> onnx_trt (FP32, FP16, INT8)
     - densenet201-onnx_trt_fp16.plan


- 김경록 M: TensorRT-JetsonNano
  - torch -> torch_trt (FP32, FP16, INT8)
    - densenet201-jn-torch_trt_fp16.pth
  - torch -> onnx -> onnx_trt (FP32, FP16, INT8)
    - densenet201-jn-onnx_trt_fp16.plan


- 김경록 M: Pruning
  - densenet201-pruning.pth


- 이장원 M: Triton Client 개발
- 이장원 M: Desktop 실험 (Flask)
- 이장원 M: Jetson Nano 실험 (Flask)
- Desktop 실험 (Triton)
- Jetson Nano 실험 (Triton)


## Environments
- Python3.6 or later
- PyTorch 1.10.1
- torchvision 0.11.2
- docker 20.10 with nvidia container toolkit
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## Models
imagenet classification Models from `torchvision 0.11.2` (https://pytorch.org/vision/0.11/models.html) 
- torchvision.models.resnet34
- torchvision.models.mobilenet_v2
- torchvision.models.efficientnet_b0
- torchvision.models.efficientnet_b7

## Data
ILSVRC2012 validation images (50,000 image) 

## Devices
- AMD 3990X ThreadRipper / Nvidia RTX 3090 24GB / RAM 128GB
- Nvidia Jetson Nano 4GB

## Model Servings
- Flask
- Nvidia Triton

## Model Compression
- TensorRT
- Torch-TensorRT
- Pruning
- Quantization

### Flask Serving
- Before Run Triton Model Serving, Check `./Flask/Models/` directory.
- if there's no `./Flask/Models/` directory, Run download_xxx_models.ipynb By Jupyter Notebook to download model files.
- Build Docker Image
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -t flask_app/torch .
```
- Run Flask Server Container
```bash
$ sudo docker run \
--rm --gpus all \
-v <host model dir>:<flask model repository> \
-p <host port>:<container port> \
--name <container name> \
flask_app/torch \
python3 flask_server.py --model-repository <flask model repository> \
--model <.pth file> --port <container port>

# Example
$ sudo docker run \
--rm --gpus all \
-v $(pwd)/Flask/Models:/Models \
-p 8000:8000 \
--name flask_app \
flask_app/torch \
python3 flask_server.py --model-repository=/Models \
--model resnet152-script.pth --port 8000
```
- Check flask_app response
```bash
$ curl -X GET "http://<ip>:<port>/model"
# Example
$ curl -X GET "http://127.0.0.1:8000/model"
# {"model":"resnet152-script.pth"}
```
- Check flask_app Process ID using `nvidia-smi`
- Run `process_monitor.py` for gpu memory usage of torch model
```bash
$ python ../process_monitor.py --target_pid <pid> --log_file <log_file>
# Example
$ python ../process_monitor.py --target_pid 31888 --log_file Flask/Monitors/resnet152-script-b1.log
```
- Run `flask_client.py`
```bash
$ python flask_client.py \
--imagenet_dir <dir> \
--log_file <log_file> \
--ip <flask_app_ip> \
--port <flask_app_port> \
--batch_size <batch_size>

# Example
$ python flask_client.py \
--imagenet_dir /home/data/ImageNet/ \
--log_file Flask/Results/resnet152-script-b1.log \
--ip localhost \
--port 8000 \
--batch_size 1
```

### Nvidia Triton Serving
- Before Run Triton Model Serving, Check `./Triton/Models/` directory.
- if there's no `./Triton/Models/` directory, Run download_xxx_models.ipynb By Jupyter Notebook to download model files.
- Check structure of `./Trition/Models/` directory and `config.pbtxt`.
- Change Directory
```bash
$ cd TCL_ModelCompression/Torch
```
- Run Triton Server Container
```bash
$ sudo docker run \
--rm --gpus all \
-v <host model dir>:<triton model repository> \
-p <host port1>:<trtion HTTP port> \
-p <host port2>:<trtion gRPC port> \
-p <host port3>:<trtion Metrics port> \
--name <container name>
nvcr.io/nvidia/tritonserver:22.08-py3 \
tritonserver --model-repository=<triton model repository> \
--model-control-mode=explicit \
--load-model=<model_name>

# Example
$ sudo docker run \
--rm --gpus all \
-v $(pwd)/Triton/Models:/Models \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
--name triton_app \
nvcr.io/nvidia/tritonserver:22.08-py3 \
tritonserver --model-repository=/Models \
--model-control-mode=explicit \
--load-model=resnet152-script
```
- Check triton app response
```bash
$ curl -X GET "http://<ip>:<port>/metrics"
# Example
$ curl -X GET "http://localhost:8002/metrics"
# Triton Metric Text...
```
- Check flask_app Process ID using `nvidia-smi`
- Run `process_monitor.py` for gpu memory usage of torch model
```bash
$ python ../process_monitor.py --target_pid <pid> --log_file <log_file>
# Example
$ python ../process_monitor.py --target_pid 31888 --log_file Triton/Monitors/resnet152-script-http-b1.log
```
- Run `triton_client.py`
```bash
$ python triton_client.py \
--imagenet_dir <dir> \
--log_file <log_file> \
--ip <triton_app_ip> \
--port <triton_app http/grpc port> \
--protocol <http/grpc> \
--batch_size <batch_size>

# Example
$ python triton_client.py \
--imagenet_dir /home/data/ImageNet/ \
--log_file Triton/Results/resnet152-script-http-b1.log \
--ip localhost \
--port 8000 \
--protocol http \
--batch_size 1
```
