
import torch
import torch.nn as nn
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class GRN(nn.Module):
    def __init__(self):
        super(GRN, self).__init__()
        # Main Encoder Part
        self.dilaconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), dilation=(1, 1), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), dilation=(1, 1), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5),  dilation=(1, 2), padding=(2, 4)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), dilation=(1, 4), padding=(2, 8)),
            nn.ELU()
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(kernel_size=1, dilation=1, out_channels=256, in_channels=5152),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.glus_0 = nn.ModuleList([GLU(2**i, 256) for i in range(6)])
        self.glus_1 = nn.ModuleList([GLU(2**i, 256) for i in range(6)])
        self.glus_2 = nn.ModuleList([GLU(2**i, 256) for i in range(6)])

        self.conv1d_3 = nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=256, in_channels=256),
                                      nn.BatchNorm1d(256),
                                      nn.ELU())
        self.conv1d_4 = nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=128, in_channels=256),
                                      nn.BatchNorm1d(128))
        # self.linear = nn.Linear(128, 128)
        self.conv1d_5 = nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=161, in_channels=128),
                                      nn.BatchNorm1d(161),
                                      nn.Sigmoid())

    def forward(self, x):
        input = x
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x = self.dilaconv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size()[0], x.size()[1], -1)
        # print(x.shape)
        x = self.conv1d(x.permute(0, 2, 1))
        # print(x.shape)
        out_list = []
        for id in range(6):
            x, out = self.glus_0[id](x)
            out_list.append(out)
        for id in range(6):
            x, out = self.glus_1[id](x)
            out_list.append(out)
        for id in range(6):
            x, out = self.glus_2[id](x)
            out_list.append(out)

        for i in range(len(out_list)):
            x = x + out_list[i]

        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        # x = x.permute(0, 2, 1)
        # x = self.linear(x)
        # x = x.permute(0, 2, 1)
        mask = self.conv1d_5(x)
        mask = mask.permute(0, 2, 1)
        return input * mask



class GLU(nn.Module):
    def __init__(self, dilation, in_channel, causal_flag=False):
        super(GLU, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=1),
            nn.BatchNorm1d(64))
        if causal_flag is True:
            self.pad = nn.ConstantPad1d((int(dilation * 6), 0), value=0.)
        else:
            self.pad = nn.ConstantPad1d((int(dilation * 3), int(dilation * 3)), value=0.)

        self.left_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=7, dilation=dilation),
            nn.BatchNorm1d(64))
        self.right_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=7, dilation=dilation),
            nn.BatchNorm1d(num_features=64),
            nn.Sigmoid())
        self.out_conv = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=1),
            nn.BatchNorm1d(256))
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        x = x1 * x2
        x = self.out_conv(x)
        out = x
        x = x + inpt
        x = self.out_elu(x)
        return x, out
