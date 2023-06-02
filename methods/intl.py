import torch
from methods.base import BaseMethod
from methods.whitening import Whitening2dIterNorm

class INTL(BaseMethod):
    # Iterative Normalization with Trace loss

    def __init__(self, cfg):
        super().__init__(cfg)
        self.IterNorm = Whitening2dIterNorm(axis=cfg.axis, iterations=cfg.iters)

    def forward(self, samples):
        loss = 0
        nmb_crops = len(samples)

        for x in samples:
            x.cuda(non_blocking=True)

        t = [self.IterNorm(self.projection(self.backbone(x))) for x in samples]
        tl = [TL(x,self.axis) for x in t]

        for i in range(1,nmb_crops):
            loss += self.loss_f(t[i], t[0]) + self.trade_off * (tl[i] + tl[0])
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
