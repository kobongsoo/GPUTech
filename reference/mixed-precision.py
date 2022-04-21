import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, *args):
    super(self, Model).__init__(*args)
    self.model = nn.Sequential(
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 20)
    )
    
  def forward(self, x):
    return self.model(x)
  
fp32_model = Model() # FP32 model
fp16_model = Model().half() # just call .half() method to obtain FP16 version of the defined model