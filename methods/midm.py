from methods.base import BaseMethod
from methods.whitening import Whitening2dIterNorm
import torch
import copy
from itertools import chain

class MIDM(BaseMethod):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ITN = Whitening2dIterNorm(dist=cfg.distributed,
                                        eps=cfg.w_eps,
                                        axis=cfg.axis,
                                        iterations=cfg.iters)

        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection = copy.deepcopy(self.projection)
        for param in chain(self.momentum_backbone.parameters(), 
                           self.momentum_projection.parameters()):
            param.requires_grad = False
        self.m = cfg.m

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

        w_q = [self.ITN(self.projection(self.backbone(x))) for x in samples[1:]]
        w_k = self.ITN(self.momentum_projection(self.momentum_backbone(samples[0])))
        c_q = [SL(x,self.axis) for x in w_q]
        
        for i in range(nmb_crops - 1):
            loss += self.loss_f(w_q[i],w_k) + self.trade_off * c_q[i]

        loss /= (nmb_crops - 1)
        return loss

def SL(x: torch.Tensor, axis) -> torch.Tensor:
    # spherical distribution loss
    if axis == 0:
        x = x.T
    N, D = x.size()
    x = x - x.mean(dim=0)
    d = torch.pow(x,2).sum(axis = 0) / (N - 1)
    sl = d.add_(-1).pow_(2).sum()
    return sl
