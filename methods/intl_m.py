from methods.base import BaseMethod
from methods.whitening import Whitening2dIterNorm
import torch
import copy
from itertools import chain

class INTL_M(BaseMethod):
    # Iterative Normalization with Trace loss (Momentum)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.IterNorm = Whitening2dIterNorm(axis=cfg.axis, iterations=cfg.iters)

        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection = copy.deepcopy(self.projection)
        for param in chain(self.momentum_backbone.parameters(), 
                           self.momentum_projection.parameters()):
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), 
                                    self.momentum_backbone.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projection.parameters(),
                                    self.momentum_projection.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
    
    def forward(self, samples):
        loss = 0
        nmb_crops = len(samples)
        with torch.no_grad():
            self._momentum_update_key_encoder(self.m)

        for x in samples:
            x.cuda(non_blocking=True)

        tq = [self.IterNorm(self.projection(self.backbone(x))) for x in samples[1:]]
        tk = self.IterNorm(self.momentum_projection(self.momentum_backbone(samples[0])))
        tl = [TL(x,self.axis) for x in tq]
        
        for i in range(nmb_crops - 1):
            loss += self.loss_f(tq[i], tk) + self.trade_off * tl[i]
        loss /= (nmb_crops - 1)
        return loss

def TL(x: torch.Tensor, axis) -> torch.Tensor:
    # Trace loss
    if axis == 0:
        x = x.T
    N, _ = x.size()
    x = x - x.mean(dim=0)
    d = torch.pow(x,2).sum(axis = 0) / (N - 1)
    tl = d.add_(-1).pow_(2).sum()
    return tl
