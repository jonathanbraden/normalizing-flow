#!/usr/bin/env python
import torch
import numpy as np

class NormalizingFlow(tensor.nn.Module):
    """
    Normalizing Flow model to map from a known to target distribution.

    Composed of a stack of individual flow layers
    """

    def __init__(self, prior, layers, target=None):
        """
        Constructor

        Input:
          prior - The prior distribution to draw from
          flows - An array storing the individual layers of the flow
        """
        super().__init__()
        self.prior = prior
        self.layers = nn.ModuleList(layers)
        self.target = target

    def flow_from_prior(self, batch_size=1):
        """
        Sample from the normalizing flow.
        The choice of prior is included in the definition of the Normalizing Flow

        Input:
          batch_size : number of samples to generate (default 1)
        """
        x, log_prob = self.prior(batch_size)
        for layer in self.layers:
            x, log_J = layer(x)
            log_prob -= logJ
        return x, log_prob

    def train(target, loss_fn, optimizer, batch_size, logging):
        optimizer.zero_grad()

        x,logq = self.flow_from_prior( batch_size = batch_size )
        logp = -target(x) # Decide if I want to keep this as an input, or part of scalar distribution
        loss = kl_divergence(logp, logq)  # replace with loss_function
        loss.backward()
        optimizer.step()

        if logging: # Add logging if necessary
            pass
        
    def save(self, fName):
        torch.save(self.state_dict(), fName)

    def load(self, fName):
        torch.load_state_dict(torch.load(fName))

# Need to implement KL Divergence
        
"""
def train_step(flow_model, target, loss_fn, optimizer, batch_size, logging):
    optimizer.zero_grad()

    x, logq = flow_model.flow_from_prior( batch_size=batch_size )
    logp = -target(x)
    loss = kl_divergence(logp, logq)
    loss.backward()

    optimizer.step()

    if logging:
        pass

    return
"""  
