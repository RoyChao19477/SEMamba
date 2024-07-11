# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/utils.py

import torch
import torch.nn as nn

class LearnableSigmoid1D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 1D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function.
    """
    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid1D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid1D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)

class LearnableSigmoid2D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 2D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function for 2D inputs.
    """
    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid2D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid2D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)
