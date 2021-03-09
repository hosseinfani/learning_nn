from torch import nn
import torch
import torch.nn.functional as F

class BengioNLM(nn.Module):
    def __init__(self, params, gpu=False):
        super().__init__()
        if gpu: self.cuda()
        else: self.cpu()

        self.params = params
        self.W_I = nn.Linear(self.params['w'] * self.params['d'], self.params['h'], bias=True)
        self.W_O = nn.Linear(self.params['h'], params['v'], bias=True)

    def forward(self, x):
        x = self.W_I(x)
        x = torch.tanh(x)
        x = self.W_O(x)
        x = torch.softmax(x, dim=1)
        return x

    def loss(self, y_, y):#y_ is the probs (1, v), y is the class index (1,1)
        #CRAZY: the default of nll_loss in torch is that it receives log of the softmax (double logs!)
        return F.nll_loss(torch.log(y_.view(1, self.params['v'])), torch.as_tensor([y]))

    def save(self, path, **kwargs):
        model = {'params': self.params,
                 'state': self.state_dict()}
        extra = '_'.join([k + str(v) for k, v in kwargs.items()])
        torch.save(model, f"{path}/{type(self).__name__}_w{self.params['w']}_d{self.params['d']}_h{self.params['h']}_v{self.params['v']}{extra}.ptc")

#BengioNLM = nn.Sequential(nn.Linear(w * d, h),
#                         nn.Tanh(),
#                         nn.Linear(h, v))
#                         nn.Softmax())
