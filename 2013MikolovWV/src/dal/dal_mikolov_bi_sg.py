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
        data = scipy.sparse.load_npz('./../data/mikolov_bi_sg_data.npz')
    except:
        titles = []
        with open(rawdata, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                #vocabs = vocabs.union(set(words))
                titles.append(words)
        stream_tokens = [token for title in titles for token in title]
        vocabs = list(dict.fromkeys(stream_tokens).keys())
        with open('./../data/vocabs.txt', 'w', encoding='utf-8') as f:
            for token in vocabs:
                f.write(f'{token}\n')


        params.wv['w'] = 2
        bigrams = [ngram for ngram in ngrams(stream_tokens, params.wv['w'])]#pair of (i, j)

        #future: n-grams to bigram pairs
        #n2bi_grams = [ngram for title in titles for ngram in ngrams(title, params.wv['w'])]

        # create batches
        params.wv['v'] = len(vocabs)
        training_size = len(bigrams)
        if sample:
            training_size = sample
        #
        #Buy more RAM!!
        #data = torch.empty(training_size, 1 * params.wv['v'] + 1)#1-hot vector + label (index of next word)

        #Sparse Matrix and bucketing
        data = sparse.lil_matrix((training_size, 1 * params.wv['v'] + 1))
        data_ = np.zeros((params.wv['b'], 1 * params.wv['v'] + 1))
        j = -1
        for i, grams in enumerate(bigrams):
            if i >= training_size: break
            X = np.zeros((1, len(vocabs)))
            X[0, vocabs.index(grams[0])] = 1
            y_index = vocabs.index(grams[1])
            y = np.asarray([y_index]).reshape((1,1))
            X_y = np.hstack([X, y])
            try:
                j += 1
                data_[j] = X_y
            except:
                s = int(((i / params.wv['b']) - 1) * params.wv['b'])
                e = int(s + params.wv['b'])
                data[s: e] = data_
                j = 0
                data_[j] = X_y
            if (i % params.wv['b'] == 0):
                print(f'Loading {i}/{len(bigrams)} instances!{datetime.now()}')

        if j > -1:
            data[-j-1:] = data_[0:j+1]

        scipy.sparse.save_npz('./../data/mikolov_bi_sg_data.npz', data.tocsr())

        #Sparse Matrix => Slow!
        # for i, grams in enumerate(bigrams):
        #     if i >= training_size: break
        #     # input
        #     X = sparse.csr_matrix((1, len(vocabs)))
        #     X[0, vocabs.index(grams[0])] = 1
        #
        #     # label: in our case, we have |V|-classifier and the class# is the index of word in the vocab
        #     y_index = vocabs.index(grams[1])
        #
        #     y = sparse.csr_matrix((1,1))
        #     y[0, 0] = y_index
        #     X_y = sparse.hstack([X, y])
        #
        #     data[i] = X_y
        #     if (i%1000 == 0):
        #         print(f'Loading {i}/{len(bigrams)} instances!')

        # don't have enough memory => bring it to batches
        # if params.lm['g']:
        #     data.cuda()


    return PrepareData(X=data[:, :-1], y=data[:, -1])

def load_vocabs():
    with open('./../data/vocabs.txt', 'r', encoding='utf-8') as f:
        vocabs = f.readlines()
    return [word.strip() for word in vocabs]

if __name__ == "__main__":
    load_data(rawdata='./../../2003BengioNLM/data/news_title_preprocess.txt', sample=None)
    print(len(load_vocabs()))