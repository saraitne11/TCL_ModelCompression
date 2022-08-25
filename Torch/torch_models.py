import torch
import torchvision
import os


def get_model(model_name: str, model_dir='./Models'):
    if model_name == 'resnet':
        model = torch.load(os.path.join(model_dir, 'ResNet152_Base.pth'))

    elif model_name == 'efficientnet':
        model = torch.load(os.path.join(model_dir, 'EfficientNet_V2_L_Base.pth'))

    elif model_name == 'densenet':
        model = torch.load(os.path.join(model_dir, 'DenseNet201_Base.pth'))

    elif model_name == 'mobilenet':
        model = torch.load(os.path.join(model_dir, 'MobileNet_V2_Base.pth'))

    else:
        raise ValueError(f"model_name {model_name} is invalid")

    return model


def get_transform(model_name: str):
    if model_name == 'resnet':
        transform = torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V1.transforms()

    elif model_name == 'efficientnet':
        transform = torchvision.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()

    elif model_name == 'densenet':
        transform = torchvision.models.densenet.DenseNet201_Weights.IMAGENET1K_V1.transforms()

    elif model_name == 'mobilenet':
        transform = torchvision.models.mobilenetv2.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    else:
        raise ValueError(f"model_name {model_name} is invalid")

    return transform
