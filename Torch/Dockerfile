FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
COPY ./Models/*.pth /app/Models/
COPY ./flask_server.py /app
COPY ./torch_models.py /app
COPY ./imagenet.json /app
RUN pip3 install flask
WORKDIR /app
