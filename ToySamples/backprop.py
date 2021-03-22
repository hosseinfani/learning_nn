import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def f1(x):
    return x

def f2(x):
    return 2 * x

def f3(x):
    return x ** 2



#with torch.enable_grad():
#requires_grad=True => only on torch.FloatTensor

x = torch.rand(*(1,1), requires_grad=True)
print(f"x: {x.data} requires_grad: {x.requires_grad} grad: {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

y1 = f1(x)
y1.backward()
print(f"x: {x.data} requires_grad: {x.requires_grad} grad (df1/dx=dx/dx=1): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")
print(f"y: {y1.data} requires_grad: {y1.requires_grad} grad: {y1.grad} grad_fn: {y1.grad_fn} is_leaf: {y1.is_leaf}")

#BE CAREFULL: it *accumulates* all the operations on x. Although y2 is different from y1, x has already f1 in its resume
y2 = f2(x)
y2.retain_grad()
y2.backward()
print(f"x: {x.data} requires_grad: {x.requires_grad} grad (df2/dx+df1/dx=d(2x)/dx+dx/dx=2+1): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")
print(f"y: {y2.data} requires_grad: {y2.requires_grad} grad: {y2.grad} grad_fn: {y2.grad_fn} is_leaf: {y2.is_leaf}")

f3(x).backward()
print(f"x: {x.data} requires_grad: {x.requires_grad} grad(df3/dx+df2/dx+df1/dx=d(x^2)/dx+d(2x)/dx+dx/dx=2x+2+1): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

#detach
x=x.detach()
x.requires_grad = True
f3(x).backward()
print(f"x.detach(): {x.data} requires_grad: {x.requires_grad} grad(df3/dx=d(x^2)=2x): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

#detach*incorrect answer*
# x.grad.detach_()
# f3(x).backward()
# print(f"x.grad.detach_(): {x.data} requires_grad: {x.requires_grad} grad(df3/dx=d(x^2)=2x): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

#x.grad.zero_()
x.grad.zero_()
f3(x).backward()
print(f"x.grad.zero_(): {x.data} requires_grad: {x.requires_grad} grad(df3/dx=d(x^2)=2x): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

x.grad.zero_()
f3(f2(f1(x))).backward()
print(f"f3(f2(f1(x))): {x.data} requires_grad: {x.requires_grad} grad(df3/df2*df2/df1*df1/dx): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")

x.grad.zero_()
y = f3(f2(f1(x)))
y.retain_grad()
y.backward()
print(f"f3(f2(f1(x))): {x.data} requires_grad: {x.requires_grad} grad(df3/df2*df2/df1*df1/dx): {x.grad} grad_fn: {x.grad_fn} is_leaf: {x.is_leaf}")
print(f"y: {y.data} requires_grad: {y.requires_grad} grad(df3/df2*df2/df1*df1/dx): {y.grad} grad_fn: {y.grad_fn} is_leaf: {y.is_leaf}")



