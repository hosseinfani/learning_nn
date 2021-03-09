import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

import params
from nn.mikolov_sg import MikolovBiSG
import dal.dal_mikolov_bi_sg as dal

def vis(e):
    model = torch.load(f'./../results/MikolovBiSG_w5_d100_v81272lr0.01_b5000_e{e}.ptc')
    mikolovbisg = MikolovBiSG(model['params'])
    mikolovbisg.load_state_dict(model['state'])

    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=10)
    Y = tsne.fit_transform(mikolovbisg.W_I.weight.detach().numpy().transpose())
    plt.scatter(Y[:, 0], Y[:, 1])

def topn(source, e):
    ##load an existing model
    model = torch.load(f'./../results/MikolovBiSG_w5_d100_v81272lr0.01_b5000_e{e}.ptc')
    mikolovbisg = MikolovBiSG(model['params'])
    mikolovbisg.load_state_dict(model['state'])
    vocabs = dal.load_vocabs()

    s_idx = vocabs.index(source)
    s_v = torch.zeros(1, len(vocabs))
    s_v[0, s_idx] = 1
    s_t = mikolovbisg.forward(s_v)#softmax(WI * WO)
    probs, predictions = s_t.topk(10, dim=1)
    # print(mikolovbisg.W_I.weight[:,s_idx], mikolovbisg.W_O.weight[s_idx, :])
    # print(mikolovbisg.W_I.weight[:,predictions[0][0]], mikolovbisg.W_O.weight[predictions[0][0], :])
    print(f'{source}:{([vocabs[t_idx] for t_idx in predictions[0]])}')

for e in range(0, 31, 5):
    topn('korea', e)

#vis(20)