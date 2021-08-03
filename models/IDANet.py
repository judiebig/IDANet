import torch
import torch.nn as nn
from typing import Any
import numpy as np
from torchvision.ops import deform_conv2d
import torch.nn.functional as F


class IDANet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(IDANet, self).__init__()
        print(f"initialize {self.__class__.__name__}")
        self.en = Encoder()
        self.de = Decoder()
        self.sab_list = nn.ModuleList([SAB(dila_rate=2 ** i, input_channel=32) for i in range(6)])
        self.att_list = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(32 * 2, 32, 1, 1, 0),
                nn.Sigmoid()
            ) for i in range(6)])

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.en(x)  # [b,32,t,f]
        c = x
        for i in range(6):
            c_pool = torch.cat([F.adaptive_avg_pool2d(c, (1, 1)), F.adaptive_max_pool2d(c, (1, 1))], dim=1)
            dis_att = self.att_list[i](c_pool)
            dis_c = c * dis_att
            c = dis_c
            x = self.sab_list[i](x, dis_c)
            c += x
        x = self.de(c)
        x = x.squeeze(dim=1)
        return x


class AttLayer(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, out_channels):
        super(AttLayer, self).__init__()
        self.out_channels = out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(out_channels, out_channels * 2, 1, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # 仿照SKNet来做
        x = x1 + x2
        s = self.global_pool(x)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(-1, 2, self.out_channels, 1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(2, dim=1))
        a_b = list(map(lambda x: x.reshape(-1, self.out_channels, 1, 1), a_b))
        output = [x1, x2]
        V = list(map(lambda x, y: x * y, output, a_b))
        out = V[0] + V[1]
        return out

class Encoder(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(Encoder, self).__init__()
        self.en1 = OperationLayer(in_channel=1, out_channel=4)
        self.en2 = OperationLayer(in_channel=4, out_channel=8)
        self.en3 = OperationLayer(in_channel=8, out_channel=16)
        self.en4 = OperationLayer(in_channel=16, out_channel=32)

    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.en4(x)
        return x


class OperationLayer(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, in_channel, out_channel, dia_late=1):
        super(OperationLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.op1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=3,
                      stride=1,
                      padding=np.int(dia_late * 2 / 2),
                      dilation=dia_late),
            nn.PReLU(self.out_channel))
        self.op2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=5,
                      stride=1,
                      padding=np.int(dia_late * 4 / 2),
                      dilation=dia_late),
            nn.PReLU(self.out_channel))
        self.op3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=7,
                      stride=1,
                      padding=np.int(dia_late * 6 / 2),
                      dilation=dia_late),
            nn.PReLU(self.out_channel))
        self.op4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=9,
                      stride=1,
                      padding=np.int(dia_late * 8 / 2),
                      dilation=dia_late),
            nn.PReLU(self.out_channel))
        self.w1 = nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.out_channel,
                            kernel_size=1,
                            stride=1)
        self.w2 = nn.Conv2d(in_channels=4 * self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=1,
                            stride=1)

    def forward(self, x):
        op1 = self.op1(x)
        op2 = self.op2(x)
        op3 = self.op3(x)
        op4 = self.op4(x)
        x_o = torch.cat([op1, op2, op3, op4], dim=1)
        x_o = self.w2(x_o)
        x_r = self.w1(x)
        x = x_o + x_r
        return x


class Decoder(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(Decoder, self).__init__()
        # multi-scale
        self.de1 = OperationLayer(in_channel=32, out_channel=16)
        self.de2 = OperationLayer(in_channel=16, out_channel=8)
        self.de3 = OperationLayer(in_channel=8, out_channel=4)
        self.de4 = OperationLayer(in_channel=4, out_channel=1)
        # offset
        self.of1 = nn.Conv2d(in_channels=16,
                             out_channels=2 * 16 * 9,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.of2 = nn.Conv2d(in_channels=8,
                             out_channels=2 * 8 * 9,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.of3 = nn.Conv2d(in_channels=4,
                             out_channels=2 * 4 * 9,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.of4 = nn.Conv2d(in_channels=1,
                             out_channels=2 * 1 * 9,
                             kernel_size=3,
                             stride=1,
                             padding=1)

        # weight
        self.w1 = nn.Parameter(torch.Tensor(16, 16, 3, 3))
        self.w2 = nn.Parameter(torch.Tensor(8, 8, 3, 3))
        self.w3 = nn.Parameter(torch.Tensor(4, 4, 3, 3))
        self.w4 = nn.Parameter(torch.Tensor(1, 1, 3, 3))
        self.b1 = nn.Parameter(torch.Tensor(16))
        self.b2 = nn.Parameter(torch.Tensor(8))
        self.b3 = nn.Parameter(torch.Tensor(4))
        self.b4 = nn.Parameter(torch.Tensor(1))

        # att
        self.att_layer1 = AttLayer(out_channels=16)
        self.att_layer2 = AttLayer(out_channels=8)
        self.att_layer3 = AttLayer(out_channels=4)
        self.att_layer4 = AttLayer(out_channels=1)

    def forward(self, x):
        x1 = self.de1(x)
        of1 = self.of1(x1)
        d1 = deform_conv2d(x1, of1, self.w1, self.b1, padding=(1, 1))
        s1 = self.att_layer1(d1, x1)
        x2 = self.de2(s1)
        of2 = self.of2(x2)
        d2 = deform_conv2d(x2, of2, self.w2, self.b2, padding=(1, 1))
        s2 = self.att_layer2(d2, x2)
        x3 = self.de3(s2)
        of3 = self.of3(x3)
        d3 = deform_conv2d(x3, of3, self.w3, self.b3, padding=(1, 1))
        s3 = self.att_layer3(d3, x3)
        x4 = self.de4(s3)
        of4 = self.of4(x4)
        d4 = deform_conv2d(x4, of4, self.w4, self.b4, padding=(1, 1))
        s4 = self.att_layer4(d4, x4)
        return s4


class SAB(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dila_rate, input_channel):
        super(SAB, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1)
        self.dila_conv_left = nn.Sequential(
            nn.PReLU(16),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=11,
                      stride=1,
                      padding=np.int((dila_rate * 10) / 2),
                      dilation=dila_rate))
        self.dila_conv_right = nn.Sequential(
            nn.PReLU(16),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=11,
                      stride=1,
                      padding=np.int((dila_rate * 10) / 2),
                      dilation=dila_rate),
            nn.Sigmoid())
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1)
        self.out_prelu = nn.PReLU(32)

    def forward(self, x, c):
        x = x + c
        input = x
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + input
        x = self.out_prelu(x)
        return x
