import numpy as np
import torch
import torch.nn as nn


inp = torch.randn(1, 10)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(inp, torch.tensor([0]))

sftmx = nn.Softmax(dim=1)
s = sftmx(inp)

for i in range(len(s[0])):
    print(-np.log(s[0][i]))