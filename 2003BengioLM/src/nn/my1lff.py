import numpy as np
import torch
import torch.nn.functional as F
class My1LFF:
  def __init__(self, params):
    self.params = params

    self.W_I = np.random.randn(params['w'] * params['d'], params['h'])
    self.B_I = np.random.randn(1, params['h'])
    print(self.W_I.shape, self.B_I.shape)

    self.W_O = np.random.randn(params['h'], params['v'])
    self.B_O = np.random.randn(1, params['v'])
    print(self.W_O.shape, self.B_O.shape)

  def forward(self, X):
    H = sigmoid(np.matmul(X, self.W_I) + self.B_I)
    O = sigmoid(np.matmul(H, self.W_O) + self.B_O)
    Y_ = softmax(O)

    # X = sigmoid(np.mm(X, W_I) + B_I)
    # X = sigmoid(np.mm(X, W_O) + B_O)
    # X = softmax(X)
    return Y_

  def __call__(self, *args, **kwargs):
    return self.forward(args[0])

def sigmoid(x):  # our activation
  return 1 / (1 + np.exp(-x))

x = np.random.randn(1)
xx = np.random.randn(2,3)

print(sigmoid(x))
print(torch.sigmoid(torch.from_numpy(x)))
print(sigmoid(xx))
print(torch.sigmoid(torch.from_numpy(xx)))

def softmax(X):
  return np.exp(X)/np.sum(np.exp(X), axis=1).reshape((X.shape[0], 1))

print(softmax(xx))
print(F.softmax(torch.from_numpy(xx), dim=1))