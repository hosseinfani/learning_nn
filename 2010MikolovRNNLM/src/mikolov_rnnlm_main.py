import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np
import scipy

import params
from nn.mikolov_rnnlm import MikolovRNNLM
import dal.dal_mikolov_rnnlm as dal

data = scipy.sparse.load_npz('./../../2013MikolovWV/data/mikolov_bi_sg_data.npz')
ds = dal.PrepareData(X=data[:, :-1], y=data[:, -1])

#do not shuffle as we have sequence here!
#We can not have batch similar to feed-forward nn due to the h_prev! We have to manully prop the update inside the training loop
ds = DataLoader(ds, batch_size=1, shuffle=False)
vocabs = dal.load_vocabs()
params.rnnlm['v'] = len(vocabs)

rnnlm = MikolovRNNLM(params.rnnlm)

criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(rnnlm.parameters(), lr=params.rnnlm['lr'])

h_prev = torch.randn(params.rnnlm['h']).view(1, params.rnnlm['h'])

e = 0
while 1:# for e in range(params.rnnlm['e']):#epochs
    running_loss = 0
    l_prev = 0
    for i, (X, y) in enumerate(ds):
        X = X.view(X.shape[0], X.shape[2])
        y = y.view(-1).type(torch.LongTensor)

        if params.rnnlm['g']:
            X = X.cuda()
            y = y.cuda()
            h_prev = h_prev.cuda()

        if i == 0:
            X_test, y_test = (X, y)
            continue

        optimizer.zero_grad()
        Y_, h_prev = rnnlm.forward(X, h_prev)
        loss = criterion(torch.log(Y_), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        rnnlm.save('./../results', lr=params.rnnlm['lr'], b=params.rnnlm['k1'], e=e)
        e += 1
        with torch.no_grad():#dynamic model? => page2, column 2 of the main paper
            Y_, h_prev = rnnlm.forward(X_test, h_prev)
            loss = criterion(torch.log(Y_), y_test)
            probs, predictions = Y_.topk(1, dim=1)
            equal = predictions == y_test.view(*predictions.shape)
            acc = torch.mean(equal.type(torch.FloatTensor))
        print(f'Epoch: {e}/{params.rnnlm["e"]}\t'
              f'Batch Loss:{running_loss/i-1}\t' 
              f'Test Loss: {loss.item()}\t'
              f'Accuracy: {acc.item() * 100}%')
        if (l_prev - loss.item() > 0):
            break


