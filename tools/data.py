from typing import Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchdyn.numerics import odeint


class Linear(nn.Module):

    def __init__(self, A: Union[None, torch.Tensor]=None) -> None:
        super().__init__()
        self.A = torch.Tensor(np.random.uniform(-1,1,size=(2,2))) if A is None else A

    def forward(self, t, y):
        return torch.mm(y, self.A)


class Cubic(nn.Module):

    def __init__(self, A: Union[None, torch.Tensor]=None) -> None:
        super().__init__()
        self.A = torch.Tensor(np.random.uniform(-1,1,size=(2,2))) if A is None else A

    def forward(self, t, y):
        return torch.mm(y**3, self.A)


class Data(Dataset):

    def __init__(self, func: nn.Module, vector_size: int = 2, data_size: int = 500, batch_time: int = 10, noise: bool = False) -> None:
        super().__init__()

        self.data_size = data_size
        self.batch_time = batch_time
        t = torch.linspace(0,25,1000)
        y0 = torch.FloatTensor(np.random.uniform(-2, 2, size=(data_size, vector_size)))

        with torch.no_grad():
            _, y = odeint(func, y0, t, solver='rk4', atol=1e-3, rtol=1e-3)

        self.y = y.transpose(0,1)

        if noise:
            self.y += 0.05*torch.randn(size=self.y.shape)

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> torch.Tensor:
        index %= self.__len__()
        tid = np.random.randint(0,len(self.y)-self.batch_time)
        return self.y[index][tid:tid+self.batch_time]
