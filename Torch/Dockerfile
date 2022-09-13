ARG BASE_IMAGE=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM $BASE_IMAGE
RUN mkdir /app
COPY ./flask_server.py /app
COPY ./imagenet.json /app
RUN pip3 install flask
WORKDIR /app
