import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
torch.manual_seed(0)

class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Arithmetic0L(nn.Module):
    def __init__(self):
        super().__init__()

        self.WI = nn.Linear(2, 1, bias=True)
        #self.WO = nn.Linear(10, 1, bias=True)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.WI(x)
        #x = F.tanh(x)
        #x = self.WO(x)
        return x

    def loss(self, y_, y):
        return self.criterion(y_, y)

class Arithmetic1L(nn.Module):
    def __init__(self):
        super().__init__()

        self.WI = nn.Linear(2, 2, bias=True)
        self.WO = nn.Linear(2, 1, bias=True)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.WI(x)
        x = F.relu(x)
        x = self.WO(x)
        return x

    def loss(self, y_, y):
        return self.criterion(y_, y)

def load_data(op):
    N = 100
    X = torch.randint(-100, 100, (N, 2)).type(torch.FloatTensor) #try integer numbers
    # X = torch.rand(N, 2)# try for real numbers in [0, 1]
    y = torch.empty(N, 1).type(torch.FloatTensor)
    if op == '+':
        y = X[:,0] + X[:,1]
    elif op == '-':
        y = X[:,0] - X[:,1]
    elif op == '*':
        y = X[:,0] * X[:,1]
    elif op == '/':
        # -59/0. = -inf == 0.7112 ==> Test Loss: inf!
        # most samples end up with 0 or +-1 since mostly the first element is less or almost equal to the second number! Need to change the training sample!
        y = torch.round(X[:,0] / X[:,1])

    X_y = torch.cat((X, y.view(N, 1)), 1)
    X_y = X_y.cuda()
    ds = PrepareData(X=X_y[:, :-1], y=X_y[:, -1])
    return DataLoader(ds, batch_size=1, shuffle=True)

epoch = 100
for op in ['+', '-', '*', '/']:
    ds = load_data(op)
    #nn_arith = Arithmetic0L()
    nn_arith = Arithmetic1L()
    nn_arith.cuda()
    optmizer = optim.Adam(nn_arith.parameters(), lr=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optmizer, factor=0.1, patience=10, mode='min')
    for e in range(epoch):
        sum_l = 0
        for i, (X, y) in enumerate(ds):
            if i == 0:
                X_test, y_test = (X, y)
                continue
            optmizer.zero_grad()
            y_ = nn_arith.forward(X)
            l = nn_arith.loss(y_, y)
            l.backward()
            optmizer.step()
            sum_l += l.item()

        else:
            with torch.no_grad():
                y_ = nn_arith.forward(X_test)
                loss = nn_arith.loss(y_, y_test)

            print(f'Epoch: {e}/{epoch}\t'
                  f'Batch Loss:{sum_l / (i - 1)}\t'
                  f'Test: ({X_test}, {op}) => {y_test} == {y_}'
                  f'Test Loss: {loss.item()}\t')
        scheduler.step(loss.item())

    print(f'{op}: WO: {nn_arith.WO.weight[0]}(tanh(WI:{nn_arith.WI.weight[0]} + b:{nn_arith.WI.bias[0]}))+ b:{nn_arith.WO.bias[0]}')
    print()
