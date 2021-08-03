import logging
import importlib
import torch
from torch.nn.utils.rnn import pad_sequence


def m_print(log):
    logging.info(log)
    print(log)


def initialize_config(module_cfg):
    """
    According to config items, load specific module dynamically with params.

    eg，config items as follow：
        module_cfg = {
            "module": "models.model",
            "main": "Model",
            "args": {...}
        }

    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def z_score(m):
    mean = torch.mean(m, [1, 2])
    std_var = torch.std(m, [1, 2])

    # size: [batch] => pad => [batch, T, F]
    mean = mean.expand(m.size()[::-1]).permute(2, 1, 0)
    std_var = std_var.expand(m.size()[::-1]).permute(2, 1, 0)

    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def pad_to_longest(batch_data):
    """
    将dataset返回的batch数据按照最长的补齐
    :param batch_data:
    :return:
    """
    mixs, cleans, lengths, name = zip(*batch_data)
    mix_list = []
    clean_list = []
    for mix in mixs:
        mix_list.append(torch.Tensor(mix))
    for clean in cleans:
        clean_list.append(torch.Tensor(clean))
    mix_list = pad_sequence(mix_list).permute(1, 0)
    clean_list = pad_sequence(clean_list).permute(1, 0)
    return mix_list, clean_list, lengths, name


def pre_pad_to_longest(batch_data):
    """
    (用于pre_train)将dataset返回的batch数据按照最长的补齐
    :param batch_data:
    :return:
    """
    snr_ori, snr_, type_ori, type_, clean_, lengths, name = zip(*batch_data)
    snr_ori_list = []
    snr_list = []
    type_ori_list = []
    type_list = []
    clean_list = []
    for ori in snr_ori:
        snr_ori_list.append(torch.Tensor(ori))
    for snr in snr_:
        snr_list.append(torch.Tensor(snr))
    for ori in type_ori:
        type_ori_list.append(torch.Tensor(ori))
    for type in type_:
        type_list.append(torch.Tensor(type))
    for clean in clean_:
        clean_list.append(torch.Tensor(clean))
    snr_ori_list = pad_sequence(snr_ori_list).permute(1, 0)
    snr_list = pad_sequence(snr_list).permute(1, 0)
    type_ori_list = pad_sequence(type_ori_list).permute(1, 0)
    type_list = pad_sequence(type_list).permute(1, 0)
    clean_list = pad_sequence(clean_list).permute(1, 0)
    return snr_ori_list, snr_list, type_ori_list, type_list, clean_list, lengths, name


def print_networks(nets: list):
    """
    计算网络参数总量
    :param nets:
    :return:
    """
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    # print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")
    return params_of_all_networks