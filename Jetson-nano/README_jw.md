### Build Docker Images
- Torch Jupyter
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Jetson-nano/Dockerfile_TorchJupyter -t torch_jupyter/jetson-nano .
```
- Flask Server
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Jetson-nano/Dockerfile_FlaskServer -t flask_server/jetson-nano .
```
- Flask Client
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Jetson-nano/Dockerfile_FlaskClient -t flask_client/jetson-nano .
```
- Tirton Client
```bash
$ cd TCL_ModelCompression/
$ sudo docker build -f Jetson-nano/Dockerfile_TritonClient -t triton_client/jetson-nano .
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
<docker image> \
jupyter notebook --allow-root \
--ip 0.0.0.0 --port <container port> \
--notebook-dir <notebook home dir> --no-browser
```
```bash
# For Script, ONNX model file
$ cd TCL_ModelCompression/
$ sudo docker run \
-d --gpus all \
-p 8881:8881 \
-v $(pwd):/TCL_ModelCompression \
-v ~/ImageNet:/ImageNet \
--name torch_jupyter \
torch_jupyter/jetson-nano \
jupyter notebook --allow-root \
--ip 0.0.0.0 --port 8881 \
--notebook-dir /TCL_ModelCompression --no-browser
```

- Check jupyter notebook URL & Token.
```bash
$ sudo docker exec torch_jupyter jupyter notebook list
```
- Open jupyter notebook and Run model file download codes.
  - `download_script_models.ipybn`
  - `download_onnx_models.ipybn`
- Run jupyter notebook codes in `ModelTest/` directory.


### Flask Serving For Desktop (Jetson-Nano)
- Before run triton model serving, Check `TCL_ModelCompression/Jetson-nano/Flask/Models/` directory.
- Create flask server container.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-p <host port>:<container port> \
-v <host repo home>:<container repo home> \
--name <container name> \
flask_server/jetson-nano \
python3 <container repository>/flask_server.py \
--model-repository <flask model repository> \
--model <.pth file> --port <container port>

# Example
$ sudo docker run \
--rm --gpus all \
-p 8000:8000 \
-v $(pwd):/TCL_ModelCompression \
--name flask_server \
flask_server/jetson-nano \
python3 /TCL_ModelCompression/flask_server.py \
--model-repository=/TCL_ModelCompression/Jetson-nano/Flask/Models \
--model resnet34-script.pt --port 8000
```
- Check flask server response.
```bash
$ curl -X GET "http://<ip>:<port>/model"
# Example
$ curl -X GET "http://127.0.0.1:8000/model"
# {"model":"resnet34-script.pt"}
```
- Run `jtop_monitor.py` for monitoring gpu memory usage of torch model.
```bash
$ cd TCL_ModelCompression/Jetson-nano/
$ sudo python jtop_monitor.py --log-file <log_file>
# Example
$ sudo python jtop_monitor.py --log-file Flask/Monitors/resnet34-script-b1.log
```
- Create flask_client container and Run `flask_client.py`.
```bash
$ cd TCL_ModelCompression/

$ sudo docker run \
--rm --gpus all \
-v <host repo home>:<container repo home> \
-v <host imagenet dir>:<container imagenet dir> \
--name flask_client \
flask_client/jetson-nano \
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
-v ~/ImageNet:/ImageNet \
--name flask_client \
flask_client/jetson-nano \
python3 /TCL_ModelCompression/flask_client.py \
--imagenet-dir /ImageNet \
--log-file /TCL_ModelCompression/Jetson-nano/Flask/Results/resnet34-script-b1.log \
--ip 10.250.72.83 \
--port 8000 \
--batch-size 1 \
--transform-dir /TCL_ModelCompression/Jetson-nano/Transforms \
--loader-workers 0
```

### Nvidia Triton Serving For Desktop (RTX 3090)
- Before Run Triton Model Serving, Check `TCL_ModelCompression/Jetson-nano/Triton/Models/` directory.
- Check structure of `TCL_ModelCompression/Jetson-nano/Trition/Models/` directory and `config.pbtxt`.
