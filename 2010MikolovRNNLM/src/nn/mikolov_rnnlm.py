from torch import nn
import torch

#Mikolov, Tomáš, et al. "Recurrent neural network based language model." Eleventh annual conference of the international speech communication association. 2010.
#It's a very special case of RNN with Truncated Backpropagation Through Time TBTT with 1 Step! => TBPTT(k1, k2=1)

#It's actually a Feedforwad NN with input [x, h_prev]
class MikolovRNNLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.U = nn.Linear(self.params['v'] + self.params['h'], self.params['h'], bias=False)
        torch.nn.init.normal_(self.U.weight, mean=0, std=0.1)
        self.V = nn.Linear(self.params['h'], self.params['v'], bias=False)
        torch.nn.init.normal_(self.V.weight, mean=0, std=0.1)

        if self.params['g']: self.cuda()
        else: self.cpu()

    def forward(self, x, h_prev):
        x = torch.cat((x, h_prev), 1)
        h_prev = self.U(x)
        x = torch.sigmoid(h_prev)
        x = self.V(x)
        x = torch.softmax(x, dim=1)
        return x, h_prev.detach()#this ignored an exception but not sure this is a fix!

    def save(self, path, **kwargs):
        model = {'params': self.params,
                 'state': self.state_dict()}
        extra = '_'.join([k + str(v) for k, v in kwargs.items()])
        torch.save(model, f"{path}/{type(self).__name__}_v{self.params['v']}{extra}.ptc")

