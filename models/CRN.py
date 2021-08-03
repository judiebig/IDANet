
import torch
import torch.nn as nn
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class CRN(nn.Module):
    def __init__(self):
        super(CRN, self).__init__()
        # Main Encoder Part
        self.en = Encoder()
        self.de = Decoder()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x, s = self.en(x)
        # print(x.shape)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size()[0], x.size()[1], -1)
        x, _ = self.lstm(x)
        x = x.reshape(x.size()[0], x.size()[1], 256, -1)
        x = x.permute(0, 2, 1, 3)
        x = self.de(x, s)
        return x.squeeze()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.en1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU()
            )
        self.en2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.en3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.en4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.en5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

    def forward(self, x):
        en_list = []
        x = self.en1(x)
        en_list.append(x)
        # print(x.size())
        x = self.en2(x)
        en_list.append(x)
        # print(x.size())
        x = self.en3(x)
        en_list.append(x)
        # print(x.size())
        x = self.en4(x)
        en_list.append(x)
        # print(x.size())
        x = self.en5(x)
        en_list.append(x)
        # print(x.size())
        return x, en_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x, x_list):
        x = self.de5(torch.cat((x, x_list[-1]), dim=1))
        # print(x.size())
        x = self.de4(torch.cat((x, x_list[-2]), dim=1))
        # print(x.size())
        x = self.de3(torch.cat((x, x_list[-3]), dim=1))
        # print(x.size())
        x = self.de2(torch.cat((x, x_list[-4]), dim=1))
        # print(x.size())
        x = self.de1(torch.cat((x, x_list[-5]), dim=1))
        # print(x.size())
        return x


