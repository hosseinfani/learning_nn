import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

wv = {
    'd': 100,#size of hidden space (same as the size of word vectors)
    'w': 2, #context window size P(i | C={1,2,..,i-1,i+1,.., w}) for all 1<=i<=w
    'v': None, ##size of vocabularly (unique tokens) and depends on vocab set
    'lr': 0.01,
    'b': 5000, #batch
    'e': 100, #epochs
    'g': 1 #gpu
}