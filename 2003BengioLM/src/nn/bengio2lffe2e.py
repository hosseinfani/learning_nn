from torch import nn
import torch
import torch.nn.functional as F
from bengio1lff_dropout import BengioLMDropout
class BengioLME2E(BengioLMDropout):
    def __init__(self, params):
        super().__init__(params)

    def forward(self, x):
        #we have to create a projection layer
        #create a matrix called w2v of size |V| * d
        #accept occurrence vector for the words happen in the context window
        #multiply the occurrence vector to the w2v such that each word selects its own vector
        return super().forward(x)
