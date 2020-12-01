import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class CoordinateInput(nn.Module):
    def __init__(self, size, num_features, learnable=False):
        super().__init__()
        
        self.size = size
        self.num_features = num_features
        self.learnable = learnable

        log_size = int(math.log(size, 2))
        x, y = torch.linspace(-1, 1, self.size), torch.linspace(-1, 1, self.size)
        grid_y, grid_x = torch.meshgrid(y, x)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)
        
        self.register_buffer("grid_xy", grid_xy)
        
        num_x = (self.num_features - 2) // 2
        num_y = (self.num_features - 2) - num_x
        fourier_x = torch.logspace(start=-3, end=log_size, base=2., steps=num_x) * 3.14
        fourier_y = torch.logspace(start=-3, end=log_size, base=2., steps=num_y) * 3.14 
        fourier_x = torch.stack([fourier_x, torch.zeros(num_x)], dim=1)
        fourier_y = torch.stack([torch.zeros(num_y), fourier_y], dim=1)
        fourier_coefficients = torch.cat([fourier_x, fourier_y], dim=0) 

        
        #Including low frequencies to approximate eucledian coordinates
        cartesian = torch.Tensor([[0.1, 0.],
                                   [0.,  0.1]])        
        fourier_coefficients = torch.cat([fourier_coefficients, cartesian], dim=0)
        assert fourier_coefficients.shape[0] == self.num_features
        
        if self.learnable:
            self.register_parameter("fourier_coefficients", fourier_coefficients)
        else:
            self.register_buffer("fourier_coefficients", fourier_coefficients)


    def forward(self, input):
        batch = input.shape[0]
        
        ff = torch.sin((self.grid_xy[:, None, :, :] * self.fourier_coefficients.T[:, :, None, None]).sum(dim=0))
        out = ff.unsqueeze(0).repeat(batch, 1, 1, 1)

        return out