from torch import nn
import torch
from torch import optim

class RNNLM(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.rnn = nn.RNN(params['v'], params['h'], num_layers=params['l'], batch_first=True, bias=False)
        self.V = nn.Linear(params['h'], params['v'], bias=False)

    def forward(self, x, h_prev):
        # it does warmup the layer by giving (x1, h0),(x2, h1),...,(x_seq, h_{seq-1})
        # so it will output h1, h2, ..., h_seq
        # so the x_seq is the same as the h_prev in the output
        x1, h_prev = self.rnn(x, h_prev)

        # it feedforward h1, h2, ..., h_seq to output layer
        x2 = self.V(x1)

        # for each output, predict what was the next word (x1 -> x2), (x2 -> x3), ...
        # the dim is very important. dim=0 is the batch#, dim=1 is the seq#, dim=2 is the output
        x3 = torch.softmax(x2, dim=2)#(batch, sequence, outputs)
        return x3, h_prev

if __name__ == "__main__":

    sentences = ['this is a test', 'they are the tests', 'this is a student']
    vocabs = dict.fromkeys([w for s in sentences for w in s.split()]).keys()
    int2word = {i: w for i, w in enumerate(vocabs)}
    word2int = {w: i for i, w in enumerate(vocabs)}
    encoded_sents_x = [[word2int[w] for w in s.split()] for s in sentences]

    one_hot_x = torch.nn.functional.one_hot(torch.tensor(encoded_sents_x), num_classes=len(vocabs)).type(torch.FloatTensor)
    one_hot_x = one_hot_x[:, :-1, :]  # up to before the last word

    encoded_sents_y = torch.tensor(encoded_sents_x)[:, 1:]

    params = {
        'v': len(vocabs), # size of input
        'h': 10,
        'l': 1, # number of recurrent layer
        's': 3, # sequence length
        'e': 100, #epoch
    }
    test_rnnlm = RNNLM(params)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(test_rnnlm.parameters(), lr=0.01)
    h = None
    for e in range(params['e']):
        optimizer.zero_grad()
        # #only check the last prediction (x3->y_==x4)
        # to do ...

        # # check each output prediction (x1->y1_==x2),(x2->y2_==x3), ...
        y_, h = test_rnnlm(one_hot_x, h)
        h = h.data#or h.detach().clone() to avoid backprop to previous sentences. Only give it to future
        loss = criterion(torch.log(y_).view(y_.shape[0] * params['s'], len(vocabs)), encoded_sents_y.reshape(y_.shape[0] * params['s']))
        loss.backward()
        optimizer.step()

        print(f'e: {e}, batch loss: {loss.item()}')

    #generate
    generate = torch.tensor(word2int['this']).view(1,1)
    h = None
    for i in range(5):
        one_hot_g = torch.nn.functional.one_hot(generate, num_classes=len(vocabs)).type(torch.FloatTensor)
        y_, h = test_rnnlm(one_hot_g, h)
        h = h.data
        probs, predictions = y_.topk(1, dim=2)#(batch, sequence, outputs)
        generate = torch.tensor(predictions.item()).view(1,1)
        print(int2word[predictions.item()])


