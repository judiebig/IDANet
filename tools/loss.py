import torch
from torch.nn.utils.rnn import pad_sequence


class MseLoss:
    @staticmethod
    def calculate_loss(ests, gths, frames):
        EPSILON = 1e-7
        masks = []
        for frame in frames:
            masks.append(torch.ones(frame, ests.size()[2], dtype=torch.float32))
        masks = pad_sequence(masks, batch_first=True).cuda()
        ests = ests * masks
        gths = gths * masks
        loss = ((ests - gths) ** 2).sum() / masks.sum() + EPSILON
        return loss


class MaeLoss:
    @staticmethod
    def calculate_loss(ests, gths, frames):
        EPSILON = 1e-7
        masks = []
        for frame in frames:
            masks.append(torch.ones(frame, gths.size()[2], dtype=torch.float32))
        masks = pad_sequence(masks, batch_first=True).cuda()
        ests = ests * masks
        gths = gths * masks
        loss = (torch.abs(ests - gths)).sum() / masks.sum() + EPSILON
        return loss

