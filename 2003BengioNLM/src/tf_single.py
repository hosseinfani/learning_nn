import numpy as np
import torch
import torch.nn.functional as F

#Bengio's LM (2003):
#Input: Sparse semantic vectors of size d, Context window size of w
#Ouput: Probabilities of all vocabs V_{1*v} conditioned on context window w, i.e., softmax
#Architecture: X_{1*d}, H_{1*h}, Y_{1*v]
#Formulation: Y = Softmax((tanh(XW_I+B_I)W_O+B_O)); W_I_{d*h}, B_I_{1*h]}, W_O_{h*v}, B_O_{1*v}

import params
from my1lff import My1LFF

params.lm['d'] = 10
params.lm['v'] = 100
lm = My1LFF(params.lm)
Y_ = lm(np.random.randn(1, params.lm['w'] * params.lm['d']))

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

corpus = ['this is a test', 'just a test for run', 'ok got it this is a test']
term_doc = CountVectorizer().fit_transform(corpus)
vocab = term_doc.get_feature_names()
params.lm['w'] = 2
params.lm['v'] = len(vocab)
params.lm['d'] = term_doc.shape[0]#term_doc_{#doc * #vocab}
lm_tf_doc = My1LFF(params.lm)
Y_lm_tf_doc = lm(term_doc[1:1+params.lm['w'], :])#Error: the vectors do not build a context window

tfidf = TfidfTransformer(smooth_idf=True,use_idf=True).fit(term_doc)
lm_tfidf = My1LFF(params.lm)
Y_lm_tfidf = lm(term_doc[1:1+params.lm['w'], :])#Error: the vectors do not build a context window

#plot the softmax
from matplotlib import pyplot as plt
plt.plot(range(1, params.lm['v'] + 1), Y_lm_tfidf)

predicted_next_word_index = np.argmax(Y_lm_tfidf)