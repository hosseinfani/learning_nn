import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

rnnlm = {
    'h': 100,#size of hidden space
    'v': None, ##size of vocabularly (unique tokens) and depends on vocab set
    'lr': 0.1,
    'k1': 1000, #after k1 timesteps (per k1 tokens), update the weights
    'e': 10, #epochs
    'g': 1 #gpu
}