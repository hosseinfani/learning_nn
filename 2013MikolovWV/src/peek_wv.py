import torch
import numpy as np

import params
from nn.mikolov_sg import MikolovBiSG
import dal.dal_mikolov_bi_sg as dal

##load an existing model
model = torch.load('./../results/MikolovBiSG_w2_d100_v81272lr0.01_b1000_e9.ptc')
mikolovbisg = MikolovBiSG(model['params'])
mikolovbisg.load_state_dict(model['state'])
vocabs = dal.load_vocabs()

def topn(source, mikolovbisg, vocabs):
    s_idx = vocabs.index(source)
    s_v = torch.zeros(1, len(vocabs))
    s_v[0, s_idx] = 1
    s_t = mikolovbisg.forward(s_v)#softmax(WI * WO)
    probs, predictions = s_t.topk(10, dim=1)
    print(f'{source}:{(probs, [vocabs[t_idx] for t_idx in predictions[0]])}')

topn(vocabs[0], mikolovbisg, vocabs)