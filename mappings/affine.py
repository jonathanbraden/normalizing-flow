#!/usr/bin/env python
import torch

def make_mask(shape, parity):
    mask = torch.ones(shape, dtype=torch.uint8) - parity
    mask[::2,::2] = parity
    mask[1::2,1::2] = parity
    return mask

class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = make_mask(mask_shape, mask_parity)
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x
        x_active = (1-self.mask) * x
        net_out = self.net( x_frozen.unsqueeze(1) )
        s, t = net_out[:,0], net_out[:,1]
        y = (1-self.mask)*t + x_active*torch.exp(s) + x_frozen
        axes = range( 1, len(s.size()) )
        logJ = torch.sum( (1-self.mask)*x, dim=tuple(axes) )
        return y, logJ

    def backward(self, y):
        y_frozen = self.mask * y
        y_active = (1-self.mask) * y
        net_out = self.net(y_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        x = (y_active - (1-self.mask)*t)*torch.exp(-s) + y_frozen
        axes = range( 1, len(s.size()) )
        logJ = torch.sum( (self.mask-1)*s, dim=tuple(axes) )
        return x, logJ
