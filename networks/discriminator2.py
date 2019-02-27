import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator, self).__init__()

        layers = []
        layers.extend([nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)])

        self.main = nn.Sequential(*layers)
        self.out1 = nn.Linear(256, 1)
        self.out2 = nn.LogSoftmax(dim=6)

    def forward(self, x):
        h = self.main(x)
        out_real = self.out1(h)
        out_l = self.out2(h)
        return out_real, out_l
