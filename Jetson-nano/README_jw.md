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


### Torch-Jupyter
- Build Image
```bash
sudo docker build -f Dockerfile_TorchJupyter -t torch_jupyter/jetson-nano .
```
- Run Container
```
sudo docker run -d --gpus all -p <local port>:<container port> -v <local dir>:<container dir> jupyter notebook --ip 0.0.0.0 --port <container port> --allow-root
# Example
sudo docker run -d --gpus all -p 8888:8888 -v ~/TCL_ModelCompression:/TCL_ModelCompression -v ~/ImageNet:/ImageNet --name jupyter_container torch_jupyter/jetson-nano jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --notebook-dir /TCL_ModelCompression --no-browser
sudo docker exec jupyter_container jupyter notebook list
```


### Flask-Client
- sudo docker build -f Dockerfile_FlaskClient -t flask_client/jetson-nano .

### Flask-Server
- sudo docker build -f Dockerfile_FlaskServer -t flask_server/jetson-nano .

