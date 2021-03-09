import torch
from torch.utils.data import Dataset, DataLoader
from nltk import ngrams
from torch import optim
import scipy
from scipy import sparse
from datetime import datetime
import numpy as np

import sys
sys.path.extend(['../src'])
import params

class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):#default of torch: float32, default of np: float64 (double)
        return torch.as_tensor(self.X[idx].toarray()).float(), torch.as_tensor(self.y[idx].toarray()).view(1,1).float()

def load_data(rawdata, sample=None):
    try:
        data = scipy.sparse.load_npz('./../data/mikolov_rnnlm_data.npz')
    except:
        titles = []
        with open(rawdata, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                #vocabs = vocabs.union(set(words))
                titles.append(words)
        stream_tokens = [token for title in titles for token in title]
        vocabs = list(set(stream_tokens))
        with open('./../data/vocabs.txt', 'w', encoding='utf-8') as f:
            for token in vocabs:
                f.write(f'{token}\n')


        params.rnnlm['v'] = len(vocabs)
        training_size = len(stream_tokens)
        if sample:
            training_size = sample

        #Sparse Matrix and bucketing
        # input data is a stream of "1-of-N" or 1-hot vectors for token each of which has the next token
        data = sparse.lil_matrix((training_size, 1 * params.rnnlm['v'] + 1))
        data_ = np.zeros((params.rnnlm['k1'], 1 * params.rnnlm['v'] + 1))
        j = -1
        for i, token in enumerate(stream_tokens[:-1]):
            if i >= training_size: break
            X = np.zeros((1, params.rnnlm['v']))
            X[0, vocabs.index(stream_tokens[i])] = 1
            y_index = vocabs.index(stream_tokens[i+1])
            y = np.asarray([y_index]).reshape((1,1))
            X_y = np.hstack([X, y])
            try:
                j += 1
                data_[j] = X_y
            except:
                s = int(((i / params.rnnlm['k1']) - 1) * params.rnnlm['k1'])
                e = int(s + params.rnnlm['k1'])
                data[s: e] = data_
                j = 0
                data_[j] = X_y
            if (i % params.rnnlm['k1'] == 0):
                print(f'Loading {i}/{len(stream_tokens)} instances!{datetime.now()}')

        if j > -1:
            data[-j-1:] = data_[0:j+1]

        scipy.sparse.save_npz('./../data/mikolov_rnnlm_data.npz', data.tocsr())

    return PrepareData(X=data[:, :-1], y=data[:, -1])

def load_vocabs():
    with open('./../data/vocabs.txt', 'r', encoding='utf-8') as f:
        vocabs = f.readlines()
    return [word.strip() for word in vocabs]

if __name__ == "__main__":
    load_data(rawdata='./../../2003BengioNLM/data/news_title_preprocess.txt', sample=None)
    print(len(load_vocabs()))