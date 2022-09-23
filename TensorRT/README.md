# Nvidia TensorRT 셋팅 방법

# 참고 자료
- Mdoel Repository 구조

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#torchscript-models



### Docker Image 실행


docker run  --gpus all -it --rm -v `local_dir:container_dir`  -p ip:port(loacl):port(docker)  nvcr.io/nvidia/pytorch:xx.xx-pyx


* Local의 TCL_ModelCompression(git_folder)와 Container의 workspace와 연동 필수


* EX)

```bash
$ docker run \
--gpus all -it --rm \
-v /home/parkys/TCL_ModelCompression/:/workspace/TCL_ModelCompression/ \
-p 10.250.73.32:8888:8888  \
nvcr.io/nvidia/pytorch:22.08-py3
```

### Jupyter notebook 실행


jupyter notebook --allow-root -ip x.x.x.x -port xxxx


* EX)

```bash
$ jupyter notebook --allow-root -ip. 0.0.0.0 -port 8888
```

*Webpage 실행 시 주의할 점

*http://`hostname:8888`/?token=849fa8170489cdcdbfddab1da890ef29f4930888a2d57df8

-> http://`10.250.73.32:8888`/?token=849fa8170489cdcdbfddab1da890ef29f4930888a2d57df8

`복사 시 hostname:port --> 10.250.73.32:8888로 변경 후 접속`



### Model Download


`1. download_Torch_TensorRT_models_fp32.ipynb`

`2. download_onnx_TensorRT_models.ipynb`

* 2개 파일 실행
* `(주의사항) : Torch_TensorRT Model의 FP16, INT8은 네트워크상의 문제가 있어서 추가 확인후 공유 예정(resnet34_fp16은 다운 가능, 이외 모델들은 불가능)`
* Onnx TensorRT Precision option(best, noTf32, fp16, int8) - Precision을 옵션을 통하여 추론속도 및 메모리 효율 향상 가능
* --best : FP32 + FP16 + INT8의 혼합정밀도
* --noTF32 : FP32 기반
* --fp16 : FP16 기반
* --int8 : INT8 기반

#### Model Respository 구조 
* Titon

`model-repository-path/
    model-name/ 
      config.pbtxt
      1/
      model.plan
`      
* Flask

`model-repository-path/
    model.plan
`
```bash

```
