#!/usr/bin/env python
import torch

class ConvNN2D(nn.Module):
    """
    Convolutional neural network with leaky ReLU nonlinearities
    """

    def __init__(self, channel_sizes, kernel_size):
        """
        """
        super().__init__()
        net = []
        padding = kernel_size//2

        for i in range(len(channel_sizes)-1):
            net.append(torch.nn.Conv2d(
                channel_sizes[i], channel_sizes[i+1], kernel_size,
                padding = padding_size, stride=1,
                padding_mode = 'circular')
                )
            if i  != len(sizes)-2:
                net.append(torch.nn.LeakyReLU())
            else:
                net.append(torch.nn.Tanh())

        self.net = torch.nn.Sequential(*net)

        # Add some code to randomize the initial weights
        
    def forward(self,x):
        return self.net(x)
