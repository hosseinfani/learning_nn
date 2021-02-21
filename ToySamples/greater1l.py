import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

class Greater1L(nn.Module):
    def __init__(self):
        super().__init__()

        self.WI = nn.Linear(2, 2, bias=True)
        self.WO = nn.Linear(2, 3, bias=True)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = self.WI(x)
        x = torch.relu(x)
        x = self.WO(x)
        x = torch.sigmoid(x)
        x = torch.softmax(x, dim=1)
        return x

    def loss(self, y_, y):
        return self.criterion(y_, y)


N = 10000
X = torch.rand(N, 2)
y = torch.empty(N, 1)
for i, x in enumerate(X):
    if x[0] > x[1]:
        y[i] = torch.as_tensor([0]).view(1, 1)
    elif x[0] < x[1]:
        y[i] = torch.as_tensor([1]).view(1, 1)
    else:
        y[i] = torch.as_tensor([2]).view(1, 1) #very rare due to random process in generating the pairs of numbers!

X_y = torch.cat((X, y), 1)

class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

ds = PrepareData(X=X_y[:, :-1], y=X_y[:, -1])
ds = DataLoader(ds, batch_size=10, shuffle=True)

greater_nn = Greater1L()
optmizer = optim.SGD(greater_nn.parameters(), lr=0.01)

epoch =100
for e in range(epoch):
    sum_l = 0
    for i, (X, y) in enumerate(ds):
        if i == 0:
            X_test, y_test = (X, y)
            continue
        optmizer.zero_grad()
        y_ = greater_nn.forward(X)
        y = y.type(torch.LongTensor)
        l = greater_nn.loss(y_, y)
        l.backward()
        optmizer.step()
        sum_l += l.item()
    else:
        with torch.no_grad():
            sum_tl = 0
            acc = 0
            y_ = greater_nn.forward(X_test)
            tl = greater_nn.loss(y_, y_test.type(torch.LongTensor))
            probs, predictions = y_.topk(1, dim=1)
            sum_tl += tl.item()
            equal = predictions == y_test.view(*predictions.shape)  # (b * 1) vs. (b)
            acc = torch.mean(equal.type(torch.FloatTensor))  # equals returns boolean values which mean function does not understand!
        print(f'Epoch: {e}/{epoch}\n'
              f'Batch Loss:{sum_l / i - 1}\n'
              f'Test Loss: {sum_tl}\n'
              f'Accuracy: {acc.item() * 100}%')


print(greater_nn.state_dict())

#try out of range numbers
y_ = greater_nn.forward(torch.as_tensor([[100., 20.]]))
print(y_.topk(1, dim=1))
