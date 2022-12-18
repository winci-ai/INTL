import torch
from methods.base import BaseMethod
from methods.whitening import Whitening2dIterNorm

class INS(BaseMethod):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ITN = Whitening2dIterNorm(dist=cfg.distributed,
                                        eps=cfg.w_eps,
                                        axis=cfg.axis,
                                        iterations=cfg.iters)

    def forward(self, samples):
        loss = 0
        nmb_crops = len(samples)

        w = [self.ITN(self.projection(self.backbone(x.cuda(non_blocking=True)))) for x in samples]
        c = [SL(x,self.axis) for x in w]

        for i in range(1,nmb_crops):
            loss += self.loss_f(w[i],w[0]) + self.trade_off*(c[i]+c[0])
        loss /= (nmb_crops - 1)
        return loss

def SL(x: torch.Tensor, axis) -> torch.Tensor:
    # Spherical loss
    if axis == 0:
        x = x.T
    N, _ = x.size()
    x = x - x.mean(dim=0)
    d = torch.pow(x,2).sum(axis = 0) / (N - 1)
    sl = d.add_(-1).pow_(2).sum()
    return sl
