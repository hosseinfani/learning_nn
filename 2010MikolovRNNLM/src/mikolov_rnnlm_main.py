import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy import sparse
import sys

import params
from nn.mikolov_rnnlm import MikolovRNNLM

data = sparse.load_npz('./../../2013MikolovWV/data/mikolov_bi_sg_data.npz')
with open('./../../2013MikolovWV/data/vocabs.txt', 'r', encoding='utf-8') as f:
    vocabs = f.readlines()
    vocabs = [word.strip() for word in vocabs]
params.rnnlm['v'] = len(vocabs)

class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):#default of torch: float32, default of np: float64 (double)
        return torch.as_tensor(self.X[idx].toarray()).float(), torch.as_tensor(self.y[idx].toarray()).view(1,1).float()

ds = PrepareData(X=data[:, :-1], y=data[:, -1])

#do not shuffle as we have sequence here!
#We can not have batch similar to feed-forward nn due to the h_prev! We have to manully prop the update inside the training loop
ds = DataLoader(ds, batch_size=1, shuffle=False)

rnnlm = MikolovRNNLM(params.rnnlm)

criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(rnnlm.parameters(), lr=params.rnnlm['lr'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False)
h_prev = torch.randn(params.rnnlm['h']).view(1, params.rnnlm['h'])

test_loss = []
running_loss = []
e = 1
while 1:# for e in range(params.rnnlm['e']):#epochs
    epoch_loss = 0
    l_prev = sys.maxsize
    for i, (X, y) in enumerate(ds):
        X = X.view(X.shape[0], X.shape[2])
        y = y.view(-1).type(torch.LongTensor)

        if params.rnnlm['g']:
            X = X.cuda()
            y = y.cuda()
            h_prev = h_prev.cuda()

        if i > 0 and i % 10000 == 0:
            X_test, y_test = (X, y)
            rnnlm.save('./../results', e=e, iter=i)
            with torch.no_grad():  # dynamic model? => page2, column 2 of the main paper
                Y_, h_prev = rnnlm.forward(X_test, h_prev)
                loss = criterion(torch.log(Y_), y_test)
                test_loss.append(loss.item())
                probs, predictions = Y_.topk(1, dim=1)
                equal = predictions == y_test.view(*predictions.shape)
                acc = torch.mean(equal.type(torch.FloatTensor))
            epoch_loss += np.average(running_loss)
            print(f'Iter: {i}\t'
                  f'Iter Loss:{np.average(running_loss)}\t'
                  f'Test Loss: {np.average(test_loss)}\t'
                  f'Accuracy: {acc.item() * 100}%')

            running_loss = []
            test_loss = []
            continue

        optimizer.zero_grad()
        Y_, h_prev = rnnlm.forward(X, h_prev)
        loss = criterion(torch.log(Y_), y)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())



    # Starting learning rate is lr = 0.1. After each epoch, the network is tested on validation data. If log-likelihood of
    # validation data increases, training continues in new epoch. If no significant improvement is observed, learning rate is halved
    # at start of each new epoch. After there is again no significant improvement, training is finished.
    # Convergence is usually achieved after 10-20 epochs. In our experiments, networks

    if epoch_loss > l_prev:
        break

    scheduler.step(epoch_loss)


