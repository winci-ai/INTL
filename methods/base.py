import torch.nn as nn
from src.model import get_backbone, get_projection
import torch.nn.functional as F
import math

class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes backbone and projection for training function.
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.out_size = get_backbone(cfg)
        self.projection = get_projection(self.out_size, cfg)
        self.emb_size = cfg.emb
        self.axis = cfg.axis
        self.loss_f = norm_mse_loss
        self.trade_off = cal_trade_off(cfg)
        self.m = 0

    def forward(self, samples):
        raise NotImplementedError

def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()

def cal_trade_off(cfg):
    if cfg.axis == 0: 
        trade_off = (math.log2(cfg.bs) - 3) * 0.01
    else:
        trade_off = (math.log2(cfg.emb) - 3) * 0.01
    return trade_off