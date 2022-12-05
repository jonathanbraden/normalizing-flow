#!/usr/bin/env python
import torch  # On MacOS have to import torch first to avoid loading two OpenMP libraries
import numpy as np
import matplotlib.pyplot as plt

def torch_mod(x):
    return torch.remainder(x, 2*np.pi)

def torch_wrap(x):
    return torch_mod(x+np.pi) - np.pi

def grab(var):
    return var.detach().cpu().numpy()

# Define a class for generating the sampling distribution
class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var) )
        self.shape = loc.shape

    def log_prob(self,x):
        logp = self.dist.log_prob(x.reshape(x.shape[0],-1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


# Helper routine to mask odd/even parity points on a square grid
def make_mask(shape, parity):
    mask = torch.ones(shape, dtype=torch.uint8) - parity
    mask[::2,::2] = parity
    mask[1::2,1::2] = parity
    return mask

# Define an affine coupling layer
class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = make_mask(mask_shape, mask_parity)
        self.net = net

    def forward(self,x):
        x_frozen = self.mask * x
        x_active = (1-self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        y = (1-self.mask)*t + x_active*torch.exp(s) + x_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum((1-self.mask)*s, dim=tuple(axes))
        return y, logJ

    def backward(self,y):
        y_frozen = self.mask * y
        y_active = (1-self.mask) * y
        net_out = self.net(y_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        x = (y_active - (1-self.mask)*t)*torch.exp(-s) + y_frozen
        axes = range(1,len(s.size()))
        logJ = torch.sum( (self.mask-1)*s, dim=tuple(axes) )
        return x, logJ
        
def make_cnn_model(*, hidden_sizes, kernel_size, in_channels, out_channels):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    padding_size = kernel_size//2
    cnn = []

    for i in range(len(sizes)-1):
        cnn.append(torch.nn.Conv2d(
            sizes[i], sizes[i+1], kernel_size, padding=padding_size,
            stride=1, padding_mode='circular'))
        if i != len(sizes)-2:
            cnn.append(torch.nn.LeakyReLU())
        else:
            cnn.append(torch.nn.Tanh())

    return torch.nn.Sequential(*cnn)
    
class ScalarAction():
    def __init__(self, m2, lam):
        self.m2 = m2
        self.lam = lam

    def __call__(self, phi):
        action = self.m2*phi**2 + self.lam*phi**4
        nDim = len(phi.shape)-1
        dims = range(1,nDim+1)
        for d_ in dims:
            action += 2*phi**2
            action -= phi*torch.roll(phi, -1, d_)
            action -= phi*torch.roll(phi,  1, d_)
        return torch.sum( action, dim=tuple(dims) )


def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq -= logJ
    return x, logq
    
def calc_dkl(logp, logq):
    return (logq-logp).mean()

def train_step(model, action, loss_fn, optimizer, batch_size, logging):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()

    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
    loss.backward()

    optimizer.step()

    # Add some logging here to track the optimizer
    return

def make_scalar_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        parity = i%2
        net = make_cnn_model( in_channels=1, out_channels=2, kernel_size=kernel_size, hidden_sizes=hidden_sizes)
        coupling = AffineCoupling(net, mask_shape=lattice_shape, mask_parity=parity)
        layers.append(coupling)
    return torch.nn.ModuleList(layers)
    
if __name__=="__main__":
    if torch.cuda.is_available():
        torch_device = 'cuda'
        float_dtype = np.float32
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch_device = 'cpu'
        float_dtype = np.float64
        torch.set_default_tensor_type(torch.DoubleTensor)

    print(f"Torch Device is {torch_device}")

    
