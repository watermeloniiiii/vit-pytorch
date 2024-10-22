import torch
import math


torch.manual_seed(111)
q = torch.randn(1, 2, 3)
k = torch.randn(1, 2, 3)
v = torch.randn(1, 2, 3)


def selfatt(dk):
    q = torch.randn(1, 2, dk)
    k = torch.randn(1, 2, dk)
    v = torch.randn(1, 2, dk)
    qk = q @ k.transpose(-2, -1)
    prob = torch.nn.functional.softmax(qk / math.sqrt(dk), dim=-1)
    return prob


print(selfatt(32))
