import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

torch.manual_seed(0)

class Bool0L(nn.Module):
    def __init__(self):
        super().__init__()

        self.WI = nn.Linear(2, 1, bias=True)
        self.criterion = nn.L1Loss()

    def forward(self, x, activation):
        x = self.WI(x)
        x = activation(x)
        return x

    def loss(self, y_, y):
        return self.criterion(y_, y)


epoch = 10000
for op in ['and', 'or', 'xor']:
    if op == 'and': training_samples = torch.as_tensor([[0., 0., 0], [0., 1., 0], [1., 0., 0], [1., 1., 1]])
    elif op == 'or': training_samples = torch.as_tensor([[0., 0., 0], [0., 1., 1], [1., 0., 1], [1., 1., 1]])
    else: training_samples = torch.as_tensor([[0., 0., 0], [0., 1., 1], [1., 0., 1], [1., 1., 0]])

    for activation in ['sigmoid', 'relu']:
        bool_nn = Bool0L()
        optmizer = optim.SGD(bool_nn.parameters(), lr=0.01)

        for e in range(epoch):
            sum_l = 0
            for i in range(training_samples.shape[0]):
                optmizer.zero_grad()
                y = training_samples[i, -1]
                y_ = bool_nn.forward(training_samples[i, :-1], torch.sigmoid if activation == 'sigmoid' else torch.relu)
                l = bool_nn.loss(y_, y)
                l.backward()
                optmizer.step()
                sum_l += l.item()
            # else:
            #     torch.no_grad()
            #     sum_tl = 0
            #     acc = 0
            #     for i in range(training_samples.shape[0]):
            #         y = training_samples[i, -1]
            #         y_ = nn_and.forward(training_samples[i, :-1])
            #         tl = nn_and.loss(y_, y)
            #         sum_tl += tl.item()
            #         acc = acc + (1 if (y_ > 0.5 and y == 1) or (y_ < 0.5 and y == 0) else 0)
            #     print(f'Epoch: {e}/{epoch}\n'
            #           f'Batch Loss:{sum_l/4}\n'
            #           f'Test Loss: {sum_tl/4}\n'
            #           f'Accuracy: {(acc/4) * 100}%')

        with torch.no_grad():
            sum_tl = 0
            acc = 0
            for i in range(training_samples.shape[0]):
                y = training_samples[i, -1]
                y_ = bool_nn.forward(training_samples[i, :-1], torch.sigmoid if activation == 'sigmoid' else torch.relu)
                tl = bool_nn.loss(y_, y)
                sum_tl += tl.item()
                acc = acc + (1 if (y_ > 0.5 and y == 1) or (y_ < 0.5 and y == 0) else 0)
        print(f'Test Loss: {sum_tl/4}\n'
              f'Accuracy: {(acc/4) * 100}%')

        print(f'{op}: {activation}(WI:{bool_nn.WI.weight[0]} + b:{bool_nn.WI.bias[0]})')
        print()
