#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt

from scalar_field import *

def test_scalar_action():
    L = 8
    lattice_shape = (L,L)

    phi_ex1 = np.random.normal(size=lattice_shape).astype(float_dtype)
    phi_ex2 = np.random.normal(size=lattice_shape).astype(float_dtype)
    cfgs = torch.from_numpy( np.stack((phi_ex1,phi_ex2), axis=0)).to(torch_device)

    print("Actions for example configs : ",ScalarAction(m2=1.,lam=1.)(cfgs))
    return
    
def test_prior_sampler():
    normal_prior = SimpleNormal(torch.zeros((3,4,5)), torch.ones((3,4,5)))
    z = normal_prior.sample_n(17)
    print(f'z.shape = {z.shape}')
    print(f'log r(z) = {grab(normal_prior.log_prob(z))}')
    return
    
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
    
    # Build a lattice model, and train
    L = 8
    lattice_shape = (L,L)
    m2, lam = 1.0, 2.0

    action = ScalarAction(m2=m2, lam=lam)
    prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))

    n_layers = 8
    hidden_sizes = [L,L]
    kernel_size = 3
    
    layers = make_scalar_affine_layers(lattice_shape=lattice_shape, n_layers=n_layers, hidden_sizes=hidden_sizes, kernel_size=kernel_size)
    model = {'layers':layers, 'prior':prior}

    base_lr = 1.e-3
    optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)

    # Training loop
    N_era = 25 # 25
    N_epoch = 100 # 100
    batch_size = 64

    import collections
    logging = collections.defaultdict(list)
    for era in range(N_era):
        for epoch in range(N_epoch):
            train_step(model, action, calc_dkl, optimizer, batch_size, logging)

        print("Done era ",era)

    torch_x, torch_logq = apply_flow_to_prior(prior, layers, batch_size=1024)
    S_eff = -grab(torch_logq)
    x = grab(torch_x)
    S = grab(action(torch_x))

    fit_intercept = np.mean(S) - np.mean(S_eff)

    fig, ax = plt.subplots()
    ax.hist2d(S_eff,S,bins=20)
    fit_intercept = np.mean(S)-np.mean(S_eff)
    xv = np.linspace(5.,35.,4,endpoint=True)
    ax.plot(xv,xv+fit_intercept,color='w')
