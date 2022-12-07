#!/usr/bin/env python
import torch
import numpy as np

class Prior:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, x):
        """
        Return the log of the probability of the sample.

        Input:
          x: Value or batch of values
        
        Output:
          log probability  of x
        """
        raise NotImplementedError

    def sample(self, batch_size=1):
        raise NotImplementedError
    

class SimpleNormal(Prior):
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var) )
        self.shape = loc.shape

    def log_prob(self,x):
        logp = self.dist.log_prob( x.reshape(x.shape[0],-1) )
        return torch.sum(logp, dim=1)

    def sample(self, batch_size=1):
        x = self.dist.sample( (batch_size,) )
        return x.reshape(batch_size, *self.shape)
