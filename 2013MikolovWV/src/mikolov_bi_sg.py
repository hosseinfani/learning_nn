import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim

import params
from nn.mikolov_sg import MikolovBiSG
import dal.dal_mikolov_bi_sg as dal

ds = dal.load_data(rawdata='./../2003BengioNLM/data/news_title_preprocess.txt', sample=None)
ds = DataLoader(ds, batch_size=params.wv['b'], shuffle=True)
vocabs = dal.load_vocabs()
params.wv['v'] = len(vocabs)

mikolovbisg = MikolovBiSG(params.wv, gpu=params.wv['g'])

criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(mikolovbisg.parameters(), lr=params.wv['lr'])

for e in range(params.wv['e']):#epochs
    running_loss = 0
    for i, (X, y) in enumerate(ds):#keep the first batch as test set :)
        #print(f'batch {i} ...')
        X = X.view(X.shape[0], X.shape[2])
        if params.wv['g']:
            X.cuda(), y.cuda()
        if i == 0:
            X_test, y_test = (X, y)
            continue
        y = y.view(-1).type(torch.LongTensor)  # PyTorch won't accept a FloatTensor as categorical target. This happened due to the way we add the last label column!
        # y = y.view(y.shape[0], 1)#1D target tensor expected, multi-target not supported

        optimizer.zero_grad()#VERY IMPORTANT
        Y_ = mikolovbisg.forward(X)#or eq. mikolovbisg(X): callable obj. instance

        loss = criterion(torch.log(Y_), y)

        # print(mikolovbisg.W_I.weight.grad)
        loss.backward()  # calculate the gardiants based on each parameter (weights)

        # print(mikolovbisg.W_I.weight.grad)
        optimizer.step()  # update the weights w_i=w_i - lr*grad

        # print(l.item())
        running_loss += loss.item()

    else:
        mikolovbisg.save('./../results', lr=params.wv['lr'], b=params.wv['b'], e=e)
        with torch.no_grad():#Very important not consider operation for test phase
            #mikolovbisg.eval() #turn off the dropout
            Y_ = mikolovbisg.forward(X_test)
            loss = criterion(torch.log(Y_), y_test.view(-1).type(torch.LongTensor))
            probs, predictions = Y_.topk(1, dim=1)
            equal = predictions == y_test.view(*predictions.shape)#(b * 1) vs. (b)
            acc = torch.mean(equal.type(torch.FloatTensor))#equals returns boolean values which mean function does not understand!
        print(f'Epoch: {e}/{params.wv["e"]}\t'
              f'Batch Loss:{running_loss/i-1}\t' 
              f'Test Loss: {loss.item()}\t'
              f'Accuracy: {acc.item() * 100}%')

        #mikolovbisg.train() #turn on the dropout back

print(f'Model architecture:{mikolovbisg}')
