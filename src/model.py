import torch.nn as nn
from torchvision import models


def get_projection(out_size, cfg):
    """ creates projection g() from config """
    x = []
    in_size = out_size
    for _ in range(cfg.projection_layers - 1):
        x.append(nn.Linear(in_size, cfg.projection_size, bias=False))
        x.append(nn.BatchNorm1d(cfg.projection_size))
        x.append(nn.ReLU(inplace=True))
        in_size = cfg.projection_size
    x.append(nn.Linear(in_size, cfg.emb))
    return nn.Sequential(*x)


def get_backbone(cfg):
    """ creates backbone E() by name and modifies it for dataset """
    zero_init = True
    if cfg.arch == 'resnet18':
        zero_init = False
    backbone = getattr(models, cfg.arch)(zero_init_residual=zero_init)
    if cfg.dataset == "cifar10" or cfg.dataset == "cifar100":
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    out_size = backbone.fc.in_features
    backbone.fc = nn.Identity()

    return backbone, out_size
