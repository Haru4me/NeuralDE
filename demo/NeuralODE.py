import argparse
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchdyn.core import NeuralODE
from torchdyn.numerics import Lorenz
import pytorch_lightning as pl
from tools.data import Linear, Cubic, Data
from tools.plots import viz2d
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--equation', type=str, choices=['linear', 'cubic', 'lorenz'], default='cubic')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
args = parser.parse_args()


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

EQUATIONS = {
    'linear': Linear(A),
    'cubic': Cubic(A)
}


class Learner(pl.LightningModule):

    def __init__(self, model: nn.Module, true_model: nn.Module):
        super().__init__()
        self.model = model
        self.true_model = true_model
        self.t = torch.linspace(0,25,1000)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        tid = np.random.randint(0,len(self.t)-10)
        y = batch
        y0 = y[:,0]
        t_eval, y_hat = self.model(y0, self.t[tid:tid+10])
        y_hat = y_hat.transpose(0,1)
        loss = F.l1_loss(y, y_hat)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        y0 = torch.Tensor([[1.5,0.]])
        t = torch.linspace(0,25,1000)

        t_eval, y_pred = self.model(y0, t)
        _, y_true = self.true_model(y0, t)
        y_true += 0.05*torch.randn(size=y_true.shape)
        viz2d(t_eval, y_true, y_pred, self.model, epoch=self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)


class LinearODEFunc(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class CubicODEFunc(LinearODEFunc):

    def __init__(self):
        super().__init__()

    def forward(self, t, y):
        return super().forward(t, y**3)


if __name__ == '__main__':

    pl.seed_everything(42)

    t = torch.linspace(0., 25., args.data_size)
    true_y0 = torch.tensor([[2., 0.]])

    if args.equation == 'linear':
        f = LinearODEFunc()
    elif args.equation == 'cubic':
        f = CubicODEFunc()

    real_f = EQUATIONS[args.equation]
    model = NeuralODE(f, sensitivity='adjoint', solver='rk4', solver_adjoint='rk4', atol_adjoint=1e-4, rtol_adjoint=1e-4)
    real_model = NeuralODE(real_f, sensitivity='adjoint', solver='rk4', solver_adjoint='rk4', atol_adjoint=1e-4, rtol_adjoint=1e-4)

    dataloader = DataLoader(Data(real_f, vector_size=2, data_size=args.data_size, batch_time=args.batch_time), batch_size=args.batch_size, num_workers=4)
    learn = Learner(model, real_model)
    trainer = pl.Trainer(max_steps=args.niters)
    trainer.fit(learn, train_dataloaders=dataloader)
