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
        self.dist = cfg.distributed
        self.loss = norm_mse_loss
        self.itn_lambda = (math.log2(cfg.bs) - 3) * 0.01

    def forward(self, samples):
        raise NotImplementedError

def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()



