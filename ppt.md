# 전사 TCL 발표 준비

## Contents

1. 왜 Edge Intelligence & Model Compression 주제를 선택했는지? 
   1. 현재 대부분 제조 환경에서는 Cloude를 활용한 중앙 집중형 AI모델 서버가 있고, N개의 데이터 수집 장치가 AI모델 서버에 요청하고 응답을 받는 아키텍쳐를 사용함
   2. 이 아키텍쳐는 AI 모델의 성능이 네트워크 환경에 따라 많이 좌지우지 됨.
   3. AI 모델의 응답 속도는 실제 모델 추론 속도 + 네트워크 응답 속도로 결정되므로, 네트워크 성능에 따라 제조라인의 생산 공정 요구 사항을 충족하지 못할 수 있음
   4. 네트워크가 지연되면 AI모델의 응답도 지연되고, 네트워크가 끊길 경우, 전체 AI모델 시스템이 중단될 수 있음
   5. 데이터 수집 장치가 늘어나면 AI모델 서버에 네트워크 병목현상이 생겨 성능이 저하될 여지도 있음

> 따라서, 제조 환경에서 실시간 공정 속도 충족 요구 사항이 있는 AI 모델 서빙 프로젝트에서는, 데이터 수집 역할을 하는 Edge Device 상에서 AI 모델을 추론을 수행할 필요가 있음.

```
Edge Device: 데이터 소스에서 직접 데이터 수집 역할을 하는 디바이스. 
Ex) 얼굴 인식 app에서 스마트폰, 제조 공장의 장비 센서  
```

2. 왜 Model Compression이 필요한지?
   1. Edge Device의 경우 Cloude 상의 서버보다 컴퓨팅 리소스가 현저히 작음. Ex) Jetson nano - ARM Coretex A57(4 Core, 1.4GHZ), 4GB RAM
   2. 컴퓨팅 리소스가 제한되는 Edge Device에서 AI 모델 추론을 수행하고 빠른 성능을 보장하기 위해, 모델 압축 과정은 필수적.
   3. 실제 제조 환경에서 Edge Device를 활용한 AI모델 배포 & 운영 프로젝트를 진행한다고 생각했을 때, 어느 Edge Device를 사용할지, 어떤 모델을 사용할지, 어떤 모델 압축 알고리즘을 사용할지에 대한 기본 설계가 중요함.
   4. 하지만 기존의 Edge Device 기반 모델 경량화 및 배포 관련 논문을 survey 해본 결과, 기본 설계를 위한 Base Line으로 활용하기 어려운 측면이 있음
      - 논문 Survey 결과 공유: 논문마다 대상 Edge Device, 경량화 알고리즘, 데이터 셋, 세부 실험 방법 등이 모두 달라, Base Line 정보로 활용하기 어려움.

> TCL 주제: Edge Device에 모델 경량화 알고리즘이 적용된 AI 모델을 서빙하고 성능 측정 실험. 

3. TCL 목적
   1. 클라우드 기반 AI 환경 (서빙 엔진, 데이터 파이프라인)을 구착하기 어려운 제조 Biz.에서 AI 모델 제안 및 아키텍처 설계를 위한 Base Line 제공
   2. 모델 경량화 방법 별로 통일된 디바이스, 데이터 셋, 측정 방법으로 실험을 진행하여 정확도, 추론 시간, 메모리 요구량 등을 산출하여 비교 분석 자료 제공
   3. 리소스가 제한되는 Edge Device 상에 AI 모델 서빙 방법(Triton, Flask) 연구
   4. TensorRT 모델 추론 최적화 프레임워크 활용 절차 내재화
   5. Pruning, Quantization 등의 모델 경량화 절차 내재화

   
4. 실험 설명
   1. 실험 구조 설명
      1. 대상 Edge Device: Desktop, Jetson nano
      2. TorchHub ImageNet Pre-training 모델 - Resnet, MobileNet, EfficientNet
      3. 모델 경량화 적용 - Onnx, TensorRT, Pruning
      4. 모델 Serving - Flask, Triton
      5. In device, Server-Process 구조
      6. Client Process
      7. Monitor Process
   2. 실험 데모 -- `동영상 녹화 필요`
      1. Jetson Nano Setup (Jetpack 설치)
      2. 기본 모델 준비
      3. 경량화 모델 준비
      4. 도커 이미지 빌드 - 서버 이미지, 클라이언트 이미지
      5. 서버 컨테이너 생성 및 서버 프로세스 실행
      6. 모니터 프로세스 실행
      7. 클라이언트 프로세스 실행
   3. 실험 결과 공유 및 설명 (Lesson Learned)
      1. 모델 추론 최적화를 위해서 TensorRT는 매우 좋은 선택임, 그러나 Nvidia 계열 디바이스에서만 지원하며, 실제 모델이 배포되는 Device에서 TensorRT 최적화를 실행해야함.
      2. ㅇㅇㅇ


5. 확장
   1. Remote Server 환경 실험
      1. Network, NAS, 1대1 연결
   2. AI 모델 LifeCycle, ScaleOut를 고려한 AI 모델 배포 아키텍처 구축 (k8s, triton 활용)
