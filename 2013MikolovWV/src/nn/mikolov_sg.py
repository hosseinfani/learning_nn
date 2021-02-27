from torch import nn
import torch
import torch.nn.functional as F

class MikolovSG(nn.Module):
    def __init__(self, params, gpu=False):
        super().__init__()
        if gpu: self.cuda()
        else: self.cpu()

        self.params = params
        self.W_I = nn.Linear(self.params['v'], self.params['d'], bias=False)#we don't have bias to make it dot product at the end!
        self.W_O = nn.Linear(self.params['d'], params['v'], bias=False)#we don't have bias to make it dot product at the end!

    def save(self, path, **kwargs):
        model = {'params': self.params,
                 'state': self.state_dict()}
        extra = '_'.join([k + str(v) for k, v in kwargs.items()])
        torch.save(model, f"{path}/{type(self).__name__}_w{self.params['w']}_d{self.params['d']}_v{self.params['v']}{extra}.ptc")

#future
class MikolovBiSG(MikolovSG):
    def __init__(self, params, gpu=False):
        super().__init__(params, gpu)

    #even may have a context of w words, we select random pair of (i, j), give i in the input and predict j in the output
    def forward(self, x):#one-hot vector, x_i = 1 => selects H = W_I[i, :], H W_O[:, j] calculates the dot product of H with all words in V
        x = self.W_I(x)
        x = self.W_O(x) #the dot products
        x = torch.softmax(x, dim=1) #only predict the one paired with i inside the bigram context (i, j)
        return x

    def loss(self, y_, y):#y_ is the probs (1, v), y is the class index (1,1)
        #CRAZY: the default of nll_loss in torch is that it receives log of the softmax (double logs!)
        return F.nll_loss(torch.log(y_.view(1, self.params['v'])), torch.as_tensor([y]))

class MikolovBiNegativeSampleSG(MikolovBiSG):
    def __init__(self, params, gpu=False):
        super().__init__(params, gpu)

    def forward(self, x):#given (i, j) that never co-occurre with each other, we want to predict -1 for j in the output
        pass

#future: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/44
class MikolovNgramSG(MikolovSG):
    def __init__(self, params, gpu=False):
        super().__init__(params, gpu)

    # given a context of w words, we select i in the middle, we predict others in the context in the output
    def forward(self, x):#one-hot vector, x_i = 1 => selects H = W_I[i, :], H W_O[:, j] calculates the dot product of H with all words in V
        x = self.W_I(x)
        x = self.W_O(x) #the dot products
        #x = torch.tanh(x) #squish the dot products in [-1, +1] if we want to do have negative sampling
        x = torch.softmax(x, dim=1) #only predict the ones inside the context
        return x

    def loss(self, y_, y):#y_ is the probs (1, v), y is the class index (1,1)
        #CRAZY: the default of nll_loss in torch is that it receives log of the softmax (double logs!)
        return F.nll_loss(torch.log(y_.view(1, self.params['v'])), torch.as_tensor([y]))


