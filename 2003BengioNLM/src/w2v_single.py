import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import params
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

titles = []
with open('./data/news_title_preprocess.txt') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        titles.append(words)

params.lm['d'] = 100
#w2v = Word2Vec(titles, min_count=1, window=params.lm['w'], size=params.lm['d'], iter=100)
#w2v.save(f'./model/w2v_model_w{params.lm["w"]}_d{params.lm["d"]}')

w2v = Word2Vec.load(f'./model/w2v_model_w{params.lm["w"]}_d{params.lm["d"]}')
w2v.init_sims(replace=True)

#sample output
word = w2v.wv.index2word[0]
print(f'{word}: {w2v.wv[word]} {w2v.wv.most_similar([word])}')
for i, y in enumerate(w2v.wv.vocab):
  if i > 10: break
  print(f'({word}, {y}): {np.dot(np.linalg.norm(w2v.wv[word]), np.linalg.norm(w2v.wv[y]))}')

#build the context vectors: given n-gram in the input, we have to concatenate their dense vectors
#1.1.) we can either create a single stream of tokens: flatten the input sensentences
#1.2.) pad each sentences to reach our desired window
#in our case, we are working with titles that may not be related! so it's better to go with second option
#2) we have to create the n-gram tokens where n+1=our window size
from nltk import ngrams
lm_n_grams = [ngram for title in titles for ngram in ngrams(title, params.lm['w'] + 1)]

#nn models
import torch
from my1lff import My1LFF
from bengio1lff import BengioLM

params.lm['v'] = len(w2v.wv.vocab)
lm_w2v = My1LFF(params.lm)
bengiolm_w2v = BengioLM(params.lm)
#we have <s> as oov since we have not learn a vector for it!
#should we learn w2v again?! Incorrectly, we can assign a random fixed vector for it.

for grams in lm_n_grams:
    print(grams)
    context = [w2v.wv[word] for word in grams]

    # label: in our case, we have |V|-classifier and the class# is the index of word in the vocab
    y_index = w2v.wv.vocab[grams[-1]].index  # the index of the true word

    # input
    X = np.asarray(context[:-1]).flatten()
    X = X.reshape((1, params.lm['w'] * params.lm['d']))
    print(f'The shape of input context vector: {X.shape}')

    # prediction by my1lff
    Y_ = lm_w2v(X)
    print(f'The shape of output prediction is: {Y_.shape}')
    print(f'We expect that {(y_index, grams[-1])} has the max prob in order to be the next word. The prob is: {Y_[0][y_index]}')
    next_word_index = np.argmax(Y_)
    max_prob = Y_[0][next_word_index]
    print(f'The nn predicts {(next_word_index, w2v.wv.index2word[next_word_index])} the next word with max prob: {max_prob}')
    nll = - np.log(Y_[0][y_index])
    print(f'The negative loglikelihood (-log p({grams[-1]})): {nll}')

    # prediction by bengio1lff
    Y_ = bengiolm_w2v(torch.from_numpy(X))
    print(f'The shape of output prediction is: {Y_.shape}')
    print(f'We expect that {(y_index, grams[-1])} has the max prob in order to be the next word. The prob is: {Y_[0][y_index]}')
    with torch.no_grad():# this is required as torch keep track of any operation on matrices for backprop
        next_word_index = np.argmax(Y_, axis=1)
        max_prob = Y_[0][next_word_index]
        print(f'The nn predicts {(next_word_index, w2v.wv.index2word[next_word_index])} the next word with max prob: {max_prob}')
        nll = - np.log(Y_[0][y_index])
        print(f'The negative loglikelihood (-log p({grams[-1]})): {nll} or {bengiolm_w2v.loss(Y_, y_index)}')

    break

#plot the softmax
#from matplotlib import pyplot as plt
#plt.plot(range(1, params.lm['v'] + 1), Y_)