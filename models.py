import torch


def load_model(model_name: str, pretrained: bool, repo='pytorch/vision'):
    model_list = torch.hub.list(repo)

    if model_name not in model_list:
        raise ValueError(f'{model_name} is invalid model')

    return torch.hub.load(repo, model_name, weights=pretrained)


def print_model_help(model_name: str, repo='pytorch/vision'):
    model_list = torch.hub.list(repo)

    if model_name not in model_list:
        raise ValueError(f'{model_name} is invalid model')

    print(torch.hub.help(repo, model_name))


if __name__ == '__main__':
    from torchinfo import summary

    _model = 'resnet101'

    # print_model_help(_model)
    mobilenet = load_model(_model, pretrained=True)
    mobilenet.cuda()

    batch_size = 32
    input_size = (batch_size, 3, 224, 224)
    summary(mobilenet, input_size)
