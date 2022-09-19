## 1. jetpack 4.6.1 base setting
 - Docker version 20.10.7
 - ubuntu18.04
 - cuda10.2
 - tensorrt and cudnn8.2.1
 - sdk 6.0

## 2. running docker with tenssort, pytorch etc

 - local folder : /home/skbluemumin/Downloads/imagenet_tar_folder
 - docker folder : /root/temp (auto mkdir)

```bash
$ docker run -it --gpus all --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /home/skbluemumin/Downloads/imagenet_tar_folder:/root/temp dustynv/jetson-inference:r32.7.1
```

```
python3 --version
# 3.6.9
```

'''
pip3 install jupyter
'''

> jupyter notebook --ip='*' --port=8888 --allow-root

```python
>>> import tensorrt
>>> tensorrt.__version__
'8.2.1.8'
>>> PyTorch 
1.10.0
>>> torchvision 
0.11.0
```


> python3 

### a. imagenet_validset_read_test.ipynb content
 - test complete

#### b. download_base_models.ipynb
 - test complete

### c. pytorch > tensorrt test
 - test complete (default setting) - efficinetnet v0 only

 - with [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

 - torch_tensorrt not installed

### d. pytorch > onmx > tensorrt test
 - not yet
