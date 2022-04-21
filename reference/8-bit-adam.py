import bitsandbytes as bnb
import torch

class Model(torch.nn.Module):
  def __init__(self):
    super(self, Model).__init__()
    self.model = torch.nn.Linear(128, 1)
    
  def forward(self, inp):
    return self.model(inp)

model = Model()

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent

# ... learning loop