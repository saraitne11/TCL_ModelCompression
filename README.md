# Model Compression Evaluation

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
- TensorRT Jupyter
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Dockerfile_TensorRTJupyter -t tensorrt_jupyter/desktop .
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
<docker image> \
jupyter notebook --allow-root \
--ip 0.0.0.0 --port <container port> \
--notebook-dir <notebook home dir> --no-browser
```
- For Script, ONNX model file
```bash
$ cd TCL_ModelCompression/
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
- For TensorRT model file
```bash
$ cd TCL_ModelCompression/
$ sudo docker run \
-d --gpus all \
-p 8881:8881 \
-v $(pwd):/TCL_ModelCompression \
-v /home/data/ImageNet:/ImageNet \
--name tensorrt_jupyter \
--shm-size 4G \
tensorrt_jupyter/desktop \
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
$ sudo docker exec [torch_jupyter|tensorrt_jupyter] jupyter notebook list
```
- Open jupyter notebook and Run model file download codes.
  - `download_script_models.ipynb`
  - `download_onnx_models.ipynb`
  - `download_onnx_tenssorrt_models.ipynb`
- Run jupyter notebook codes in `ModelTest/` directory.


### Flask Serving For Desktop (RTX 3090)
- Before run triton model serving, Check `TCL_ModelCompression/Flask/Models/` directory.
- Create flask server container.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-p <host port>:<container port> \
-v <host repo home>:<container repo home> \
--name <container name> \
flask_server/desktop \
python3 <container repository>/flask_server.py \
--model-repository <flask model repository> \
--model <.pth file> --port <container port>

# Example
$ sudo docker run \
--rm --gpus all \
-p 8000:8000 \
-v $(pwd):/TCL_ModelCompression \
--name flask_server \
flask_server/desktop \
python3 /TCL_ModelCompression/flask_server.py \
--model-repository=/TCL_ModelCompression/Flask/Models \
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
$ cd TCL_ModelCompression/
$ sudo python3 process_monitor.py --target-pid <pid> --log-file <log_file>
# Example
$ sudo python3 process_monitor.py --target-pid 31888 --log-file Flask/Monitors/resnet34-script-b1.log
```
- Create flask_client container and Run `flask_client.py`.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v <host repo home>:<container repo home> \
-v <host imagenet dir>:<container imagenet dir> \
--name flask_client \
--shm-size 4G \
flask_client/desktop \
python3 <container repository>/flask_client.py \
--imagenet-dir <container imagenet dir> \
--log-file <client log file path> \
--ip <host ip address> \
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
python3 /TCL_ModelCompression/flask_client.py \
--imagenet-dir /ImageNet \
--log-file /TCL_ModelCompression/Flask/Results/resnet34-script-b1.log \
--ip 10.250.73.32 \
--port 8000 \
--batch-size 1 \
--transform-dir /TCL_ModelCompression/Transforms \
--loader-workers 2
```

### Nvidia Triton Serving For Desktop (RTX 3090)
- Before Run Triton Model Serving, Check `TCL_ModelCompression/Triton/Models/` directory.
- Check structure of `TCL_ModelCompression/Trition/Models/` directory and `config.pbtxt`.

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
$ cd TCL_ModelCompression/
$ sudo python3 process_monitor.py --target-pid <pid> --log-file <log file>
# Example
$ sudo python3 process_monitor.py --target-pid 31888 --log-file Triton/Monitors/resnet34-script-grpc-b1.log
```
- Create triton_client container and Run `triton_client.py`.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v <host repo home>:<container repo home> \
-v <host imagenet dir>:<container imagenet dir> \
--name triton_client \
--shm-size 4G \
triton_client/desktop \
python3 <container repo home>/triton_client.py \
--imagenet-dir <container imagenet dir> \
--log-file <client log file path> \
--ip <host ip address> \
--port <triton server port> \
--protocol <http or grpc> \
--batch-size <batch size> \
--transform-dir <transform file dir> \
--loader-workers <loader workers>

# Example
$ sudo docker run \
--rm --gpus all \
-v $(pwd):/TCL_ModelCompression \
-v /home/data/ImageNet:/ImageNet \
--name triton_client \
--shm-size 4G \
triton_client/desktop \
python3 /TCL_ModelCompression/triton_client.py \
--imagenet-dir /ImageNet \
--log-file /TCL_ModelCompression/Triton/Results/resnet34-script-grpc-b1.log \
--ip 10.250.73.32 \
--port 8001 \
--protocol grpc \
--batch-size 1 \
--transform-dir /TCL_ModelCompression/Transforms \
--loader-workers 2
```

## 참고 자료
- <a href=https://github.com/leejinho610/TRT_Triton_HandsOn>220421 Nvidia AI Developer Meetup Hands-on</a>
- <a href=https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt>Nvidia TensorRT Docker Images</a>
- <a href=https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt>Nvidia TenosrRT Docker Images for Jetson</a>
- <a href=https://github.com/pytorch/TensorRT#compiling-torch-tensorrt>Torch-TensorRT</a>
- <a href=https://github.com/triton-inference-server/server/tree/main/docs>Nvidia Triton Server Github</a>
- <a href=https://github.com/triton-inference-server/server/releases>Nvidia Triton Server Realese Note</a>
- <a href=https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver>Nvidia Triton Server NGC(Nvidia Gpu Cloud)</a>
- <a href=https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb>TensorRT Inference in Python</a>
