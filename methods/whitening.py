import abc
import torch
import torch.nn as nn
import torch.distributed as dist

class Whitening2d(nn.Module):
    def __init__(self,
                 dist=False,
                 eps=0,
                 axis=0,
                 iterations=4):
        super(Whitening2d, self).__init__()
        self.eps = eps
        self.dist = dist
        self.axis = axis
        self.iterations = iterations

    def forward(self, x):
        assert self.axis in (0,1), "axis must be in (channel,batch) !"
        if self.dist:
            x = torch.cat(FullGatherLayer.apply(x), dim=0)
        
        w_dim = x.size(-1)
        m = x.mean(0 if self.axis == 1 else 1)
        m = m.view(1, -1) if self.axis == 1 else m.view(-1, 1)
        xn = x - m 

        if self.axis == 1:
            eye = torch.eye(w_dim).type(xn.type()).reshape(1, w_dim, w_dim).repeat(1, 1, 1)
            xn_g = xn.reshape(-1, 1, w_dim).permute(1, 0, 2)
        else:
            eye = torch.eye(x.size(0)).type(xn.type()).reshape(1, x.size(0), x.size(0)).repeat(1, 1, 1)
            xn_g = xn.reshape(-1, 1, w_dim).permute(1, 2, 0)

        f_cov = torch.bmm(xn_g.permute(0, 2, 1), xn_g) / (xn_g.shape[1] - 1) 
        sigma = (1 - self.eps) * f_cov + self.eps * eye

        matrix = self.whiten_matrix(sigma, eye)  
        decorrelated = torch.bmm(xn_g, matrix)

        if self.axis == 1:
            decorrelated = decorrelated.permute(1, 0, 2).reshape(-1, w_dim)
        else:
            decorrelated = decorrelated.permute(2, 0, 1).reshape(-1, w_dim)

        return decorrelated

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye):
        pass

    def extra_repr(self):
        return "distributed={}, eps={}, axis={}, iterations={}".format(
            self.dist, self.eps, self.axis, self.iterations
        )

class Whitening2dIterNorm(Whitening2d):

    def whiten_matrix(self, sigma, eye):
        trace = sigma.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        trace = trace.reshape(sigma.size(0), 1, 1)
        sigma_norm = sigma * trace.reciprocal()

        projection = eye
        for k in range(self.iterations):
            projection = torch.baddbmm(projection, torch.matrix_power(projection, 3), sigma_norm, beta=1.5, alpha=-0.5)
        wm = projection.mul_(trace.reciprocal().sqrt())
        return wm

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]