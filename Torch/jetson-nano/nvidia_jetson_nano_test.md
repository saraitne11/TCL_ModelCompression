1. jetpack 4.6.1 base setting

Docker version 20.10.7

ubuntu18.04

cuda10.2

tensorrt and cudnn8.2.1

sdk 6.0

<br/>

2. running docker with tensorrt imported

sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-tensorrt:r8.0.1-runtime

a. install pytorch, vision

pip3 install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

(cf. issue with pip3 install jupyter >> using vscode py file and debugging)


python3 --version

- 3.8.0

>> python3 

import tensorrt

tensorrt.__version__

- '8.0.1.6'

PyTorch 

- 1.10.1

torchvision 

- 0.11.2

docker commit [current_container] imagenet-test:0.21

<br/>

3. docker with imagenet validation connect

local folder : /home/skbluemumin/Downloads/imagenet_tar_folder

docker folder : /root/temp (auto mkdir)

docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /home/skbluemumin/Downloads/imagenet_tar_folder:/root/temp imagenet-test:0.21

>> python3 

a. imagenet_validset_read_test.ipynb content 

- test complete

b. download_base_models.ipynb 

- test partially complete

   (not connecting cuda:0?)

c. pytorch > tensorrt test

- not yet

d. pytorch > onmx > tensorrt test

- not yet