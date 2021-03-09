from torch import nn
import torch
import torch.nn.functional as F
from bengio1lff import BengioNLM
class BengioNLMDropout(BengioNLM):
    def __init__(self, params):
        super().__init__(params)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.W_I(x)
        x = torch.tanh(x)
        x = self.dropout(x)#only this!
        x = self.W_O(x)
        x = torch.softmax(x, dim=1)#we don't apply on our output because we need all classes!
        return x

#BengioNLM = nn.Sequential(nn.Linear(w * d, h),
#                         nn.Tanh(),
#                         nn.Dropout(p=0.2),
#                         nn.Linear(h, v))
#                         nn.Softmax())
