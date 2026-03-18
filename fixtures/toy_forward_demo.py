"""最小示例：含 nn.Module.forward，便于验证「以 forward 为入口」的调用链展示。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sub(nn.Module):
    def forward(self, x):
        return F.relu(x)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sub = Sub()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        h = self.lin(x)
        h = self.sub(h)
        return torch.softmax(h, dim=-1)

