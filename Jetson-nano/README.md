cf) [triton 관련 속도 개선 요약 테크 블로그 내용](https://tech.kakaopay.com/post/model-serving-framework/)

작업 추천 방식 : ipynb 파일 대신, python3 실행 방식 추천

-> ipynb를 쓰기 위해서 jupyter를 키면, 모델 하나만 올려도 jetson_nano가 멈춤

## 1. jetpack 4.6.1 base setting
 - Docker version 20.10.7
 - ubuntu18.04
 - cuda10.2
 - tensorrt and cudnn8.2.1
 - sdk 6.0

## 2. running docker with tenssort, pytorch etc

 - local folder : /home/skbluemumin/Downloads/imagenet_tar_folder
 - docker folder : /root/temp (auto mkdir)


AS-IS : pytorch + tensorrt + gpu 지원이 동시에 가능하였어야 하나 gpu 인식이 안됨

TO-BE : [pytorch 블로그](https://pytorch.org/blog/running-pytorch-models-on-jetson-nano/)에서 신규 이미지 파일 다운로드


<br/>

```bash
$ docker run -it --gpus all --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /home/skbluemumin/Downloads/imagenet_tar_folder:/root/temp dustynv/jetson-inference:r32.7.1
```

```
python3 --version
# 3.6.9
```


> pip3 install jupyter


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

```bash
tensorrt 변환시 int8 수행 문제

 - fp16, int8 관련 [블로그](https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/usage/reduced_precision.html)

 -> jetson nano가 int8을 지원하지 않음

  cf) [in8 관련 문의](https://forums.developer.nvidia.com/t/why-jetson-nano-not-support-int8/84060/2)
```

### a. imagenet_validset_read_test.ipynb content
 - test complete

#### b. download_base_models.ipynb
 - test complete

#### c. download_onnx_models.ipynb

 - test complete

 - efficientnet_b7은 수행시 메모리 문제로 파일이 나오지 않음(해결 방법 확인 중)

### d. pytorch > tensorrt test
 - test complete (default setting) - efficinetnet v0 only

 - torch_tensorrt not installed / with [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

 -> triton에서 torch_trt가 안되므로 시간이 되면 수행

### e. pytorch > onmx > tensorrt test

 - onnx 변환 관련 [github](https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb)

 - download_onnx_tensorrt_models.ipynb 수행