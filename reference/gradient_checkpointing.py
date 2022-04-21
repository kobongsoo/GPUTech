import torch
import torch.nn as nn
import torch.utils.checkpoint

class Layer(nn.Sequential):
  def __init__(self, *args, **kwargs):
      super(self, Layer).__init__(*args)
  
  def forward(self, *args):
    return super().forward(*args)
  
class CheckPointed(nn.Sequential):
  def forward(self, *args):
    return torch.utils.checkpoint.checkpoint(super().forward, *args)

model = nn.Sequential(
  CheckPointed(
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU())
  ),
  CheckPointed(
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU())
  )
)