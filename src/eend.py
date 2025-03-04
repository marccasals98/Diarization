import torch
from torch import nn 

class EEND_Model(nn.Module):
    def __init__(self, params, device) -> None:
        super().__init__()
        self.device = device
        # Instantiate the linear layer here so it becomes a registered parameter.
        # Assuming you know the input dimension (for example, params.input_dim)
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        print("we print x:size", x.size())
        out = self.linear(x)
        return out
