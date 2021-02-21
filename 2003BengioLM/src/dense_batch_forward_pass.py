import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from nltk import ngrams
from gensim.models import Word2Vec

import params

titles = []
with open('./data/news_title_preprocess.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        titles.append(words)

params.lm['d'] = 100
w2v = Word2Vec.load(f'./model/w2v_model_w{params.lm["w"]}_d{params.lm["d"]}')
w2v.init_sims(replace=True)

lm_n_grams = [ngram for title in titles for ngram in ngrams(title, params.lm['w'] + 1)]

# create batches
#training_size = 10000
training_size = len(lm_n_grams)
data = torch.empty(training_size, params.lm['w'] * params.lm['d'] + 1)#context vector + label (index of next word)

for i, grams in enumerate(lm_n_grams):
    if i >= training_size: break
    context = [w2v.wv[word] for word in grams]
    # label: in our case, we have |V|-classifier and the class# is the index of word in the vocab
    y_index = w2v.wv.vocab[grams[-1]].index  # the index of the true word

    # input
    X = torch.as_tensor(context[:-1]).view(1, params.lm['w'] * params.lm['d'])
    y = torch.as_tensor([y_index]).view(1, 1)
    X_y = torch.cat((X, y), 1)
    data[i] = X_y
    if (i%1000 == 0):
        print(f'Loading {i}/{len(lm_n_grams)} instances!')

# don't have enough memory => bring it to batches
# if params.lm['g']:
#     data.cuda()

from torch.utils.data import Dataset, DataLoader
class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = PrepareData(X=data[:, :-1], y=data[:, -1])
ds = DataLoader(ds, batch_size=params.lm['b'], shuffle=True)

from src.nn.bengio1lff import BengioLM
params.lm['v'] = len(w2v.wv.vocab)

##load an existing model
# model = torch.load('./results/BengioLM_w5_d100_h100_v81272lr0.01_b1000_e5.ptc')
# bengiolm_w2v = BengioLM(model['params'])
# bengiolm_w2v.load_state_dict(model['state'])

bengiolm_w2v = BengioLM(params.lm, gpu=params.lm['g'])

#from src.nn.bengio1lff_dropout import BengioLMDropout
#bengiolm_w2v = BengioLMDropout(params.lm)

from torch import optim
criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(bengiolm_w2v.parameters(), lr=params.lm['lr'])

for e in range(params.lm['e']):#epochs
    running_loss = 0
    for i, (X, y) in enumerate(ds):#keep the first batch as test set :)
        if params.lm['g']:
            X.cuda(), y.cuda()
        if i == 0:
            X_test, y_test = (X, y)
            continue
        y = y.type(torch.LongTensor)  # PyTorch won't accept a FloatTensor as categorical target. This happened due to the way we add the last label column!
        # y = y.view(y.shape[0], 1)#1D target tensor expected, multi-target not supported

        optimizer.zero_grad()#VERY IMPORTANT
        Y_ = bengiolm_w2v.forward(X)#or eq. bengiolm_w2v(X): callable obj. instance

        loss = criterion(torch.log(Y_), y)

        # print(bengiolm_w2v.W_I.weight.grad)
        loss.backward()  # calculate the gardiants based on each parameter (weights)

        # print(bengiolm_w2v.W_I.weight.grad)
        optimizer.step()  # update the weights w_i=w_i - lr*grad

        # print(l.item())
        running_loss += loss.item()

    else:
        bengiolm_w2v.save('./results', lr=params.lm['lr'], b=params.lm['b'], e=e)
        with torch.no_grad():#Very important not consider operation for test phase
            #bengiolm_w2v.eval() #turn off the dropout
            Y_ = bengiolm_w2v.forward(X_test)
            loss = criterion(torch.log(Y_), y_test.type(torch.LongTensor))
            probs, predictions = Y_.topk(1, dim=1)
            equal = predictions == y_test.view(*predictions.shape)#(b * 1) vs. (b)
            acc = torch.mean(equal.type(torch.FloatTensor))#equals returns boolean values which mean function does not understand!
        print(f'Epoch: {e}/{params.lm["e"]}\n'
              f'Batch Loss:{running_loss/i-1}\n' 
              f'Test Loss: {loss.item()}\n'
              f'Accuracy: {acc.item() * 100}%')

        #bengiolm_w2v.train() #turn on the dropout back
print(f'Model architecture:{bengiolm_w2v}')

#model save
#model load