# Model Compression Evaluation

# 참고 자료
- <a href=https://github.com/leejinho610/TRT_Triton_HandsOn>220421 Nvidia AI Developer Meetup Hands-on</a>
- <a href=https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt>Nvidia TensorRT Images</a>
- <a href=https://github.com/pytorch/TensorRT#compiling-torch-tensorrt>Torch-TensorRT</a>
- <a href=https://github.com/triton-inference-server/server/tree/main/docs>Nvidia Triton Server</a>

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
- docker 20.10 with <a href=https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>nvidia container toolkit</a>

## Models
<a href=https://pytorch.org/vision/0.11/models.html>imagenet classification Models</a> from `torchvision 0.11.2` 
- torchvision.models.resnet34
- torchvision.models.mobilenet_v2
- torchvision.models.efficientnet_b0
- torchvision.models.efficientnet_b7

## Data
ILSVRC2012 validation images (50,000 image) 

## Devices
- AMD 3990X ThreadRipper / Nvidia RTX 3090 24GB / RAM 128GB
- <a href=https://github.com/saraitne11/TCL_ModelCompression/tree/main/Jetson-nano>Nvidia Jetson Nano 4GB</a>

## Model Servings
- Flask
- Nvidia Triton

## Model Compression
- TensorRT
- Torch-TensorRT
- Pruning
- Quantization


### Build Docker Images
- Torch Jupyter
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Dockerfile_TorchJupyter -t torch_jupyter/desktop .
```
- Flask Server
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Dockerfile_FlaskServer -t flask_server/desktop .
```
- Flask Client
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Dockerfile_FlaskClient -t flask_client/desktop .
```
- Tirton Client
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Dockerfile_TritonClient -t triton_client/desktop .
```


### Download Model Files & Model Local Test
- Create docker container.
- When write command in multiple lines with "\\", there should be no characters after "\\". 
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \ 
-d --gpus all \
-p <host port>:<container port> \
-v <host dir>:<container dir> \
--name torch_jupyter \
--shm-size 4G \
torch_jupyter/desktop \
jupyter notebook --allow-root \
--ip 0.0.0.0 --port <container port> \
--notebook-dir <notebook home dir> --no-browser

# Example
$ sudo docker run \
-d --gpus all \
-p 8881:8881 \
-v $(pwd):/TCL_ModelCompression \
-v /home/data/ImageNet:/ImageNet \
--name torch_jupyter \
--shm-size 4G \
torch_jupyter/desktop \
jupyter notebook --allow-root \
--ip 0.0.0.0 --port 8881 \
--notebook-dir /TCL_ModelCompression --no-browser
```

- `--shm-size 4G` means shared memory with host of container. You can check using `df -h` command in container.
```
Filesystem      Size  Used Avail Use% Mounted on
overlay         457G  276G  158G  64% /
tmpfs            64M     0   64M   0% /dev
shm             4.0G     0  4.0G   0% /dev/shm
/dev/nvme0n1p3  457G  276G  158G  64% /ImageNet
tmpfs            63G   12K   63G   1% /proc/driver/nvidia
udev             63G     0   63G   0% /dev/nvidia0
tmpfs            63G     0   63G   0% /proc/asound
tmpfs            63G     0   63G   0% /proc/acpi
tmpfs            63G     0   63G   0% /proc/scsi
tmpfs            63G     0   63G   0% /sys/firmware
```

- Check jupyter notebook URL & Token.
```bash
$ sudo docker exec torch_jupyter jupyter notebook list
```
- Open jupyter notebook and Run model file download codes.
  - `download_script_models.ipybn`
  - `download_onnx_models.ipybn`
- Run jupyter notebook codes in `ModelTest/` directory.


### Flask Serving For Desktop(RTX 3090)
- Before run triton model serving, Check `./Flask/Models/` directory.
- Create flask server container.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-p <host port>:<container port> \
-v <host model dir>:<flask model repository> \
--name <container name> \
flask_server/desktop \
python3 flask_server.py \
--model-repository <flask model repository> \
--model <.pth file> --port <container port>

# Example
$ sudo docker run \
--rm --gpus all \
-p 8000:8000 \
-v $(pwd)/Flask/Models:/Models \
--name flask_server \
flask_server/desktop \
python3 flask_server.py \
--model-repository=/Models \
--model resnet34-script.pt --port 8000
```
- Check flask server response.
```bash
$ curl -X GET "http://<ip>:<port>/model"
# Example
$ curl -X GET "http://127.0.0.1:8000/model"
# {"model":"resnet34-script.pt"}
```
- Check `flask_server.py` PID using `nvidia-smi`.
- Run `process_monitor.py` for monitoring gpu memory usage of torch model.
```bash
$ cd TCL_ModelCompression
$ sudo python process_monitor.py --target-pid <pid> --log-file <log_file>
# Example
$ sudo python process_monitor.py --target-pid 31888 --log-file Flask/Monitors/resnet34-script-b1.log
```
- Create flask_client container and Run `flask_client.py`.
```
sudo docker run --rm --gpus all -v $(pwd):/Jetson-nano -v ~/ImageNet:/ImageNet --name flask_client flask_client/jetson-nano python3 flask_client.py --imagenet-dir /ImageNet --log-file /Jetson-nano/Flask/Results/resnet34-script-b1.log --ip 10.250.72.83 --port 8000 --batch-size 1 --transform-dir /Jetson-nano/Transforms --loader-workers 0
```
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v <host repository>:<container repository> \
-v <host imagenet dir>:<container imagenet dir> \
--name flask_client \
--shm-size 4G \
flask_client/desktop \
python3 flask_client.py \
--imagenet-dir <container imagenet dir> \
--log-file <client log file path> \
--ip <host ip address, not localhost or 127.0.0.1> \
--port <flask server port> \
--batch-size <batch size> \
--transform-dir <transform file dir> \
--loader-workers <loader workers>

# Example
$ sudo docker run \
--rm --gpus all \
-v $(pwd):/TCL_ModelCompression \
-v /home/data/ImageNet:/ImageNet \
--name flask_client \
--shm-size 4G \
flask_client/desktop \
python3 flask_client.py \
--imagenet-dir /ImageNet \
--log-file /TCL_ModelCompression/Flask/Results/resnet34-script-b1.log \
--ip 10.250.73.32 \
--port 8000 \
--batch-size 1 \
--transform-dir /TCL_ModelCompression/Transforms \
--loader-workers 2
```

### Nvidia Triton Serving For Desktop(RTX 3090)
- Before Run Triton Model Serving, Check `./Triton/Models/` directory.
- Check structure of `./Trition/Models/` directory and `config.pbtxt`.

- Create triton server container.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v <host model dir>:<triton model repository> \
-p <host port1>:<trtion HTTP port> \
-p <host port2>:<trtion gRPC port> \
-p <host port3>:<trtion Metrics port> \
--name <container name> \
nvcr.io/nvidia/tritonserver:22.08-py3 \
tritonserver \
--model-repository=<triton model repository> \
--model-control-mode=explicit \
--load-model=<model_name>

# Example
$ sudo docker run \
--rm --gpus all \
-v $(pwd)/Triton/Models:/Models \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
--name triton_server \
nvcr.io/nvidia/tritonserver:22.08-py3 \
tritonserver \
--model-repository=/Models \
--model-control-mode=explicit \
--load-model=resnet34-script
```
- Check triton server response
```bash
$ curl -X GET "http://<ip>:<port>/metrics"
# Example
$ curl -X GET "http://localhost:8002/metrics"
# Triton Metric Text...
```
- Check `triton_server` PID using `nvidia-smi`.
- Run `process_monitor.py` for monitoring gpu memory usage of torch model.
```bash
$ sudo python process_monitor.py --target_pid <pid> --log_file <log_file>
# Example
$ sudo python process_monitor.py --target_pid 31888 --log_file Triton/Monitors/resnet34-script-http-b1.log
```
- Create triton_client container and Run `triton_client.py`.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v $(pwd):/TCL_ModelCompression \
-v /home/data/ImageNet:/ImageNet \
--name triton_client \
--shm-size 4G \
triton_client/desktop \
python3 triton_client.py \
--imagenet-dir /ImageNet \
--log-file /TCL_ModelCompression/Triton/Results/resnet34-script-http-b1.log \
--ip 10.250.73.32 \
--port 8000 \
--protocol http \
--batch-size 1 \
--transform-dir /TCL_ModelCompression/Transforms \
--loader-workers 2

# Example
$ sudo docker run \
--rm --gpus all \
-v <host repository>:<container repository> \
-v <host imagenet dir>:<container imagenet dir> \
--name triton_client \
--shm-size 4G \
triton_client/desktop \
python3 triton_client.py \
--imagenet-dir <container imagenet dir> \
--log-file <client log file path> \
--ip <host ip address, not localhost or 127.0.0.1> \
--port <triton server port> \
--protocol <http or grpc> \
--batch-size <batch size> \
--transform-dir <transform file dir> \
--loader-workers <loader workers>
```
