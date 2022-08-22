import torchvision


def get_model(model_name: str):
    if model_name == 'resnet':
        transform = torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V1.transforms()
        w = torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet152(weights=w)

    elif model_name == 'efficientnet':
        transform = torchvision.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
        w = torchvision.models.efficientnet.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_v2_l(weights=w)

    elif model_name == 'densenet':
        transform = torchvision.models.densenet.DenseNet201_Weights.IMAGENET1K_V1.transforms()
        w = torchvision.models.densenet.DenseNet201_Weights.IMAGENET1K_V1
        model = torchvision.models.densenet201(weights=w)

    elif model_name == 'mobilenet':
        transform = torchvision.models.mobilenetv2.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
        w = torchvision.models.mobilenetv2.MobileNet_V2_Weights.IMAGENET1K_V1
        model = torchvision.models.mobilenet_v2(weights=w)
    else:
        raise ValueError(f"model_name {model_name} is invalid")
    categories = w.meta['categories']

    return model, transform, categories
