"""
Nonspeech dataset from Ohio State University
"""

import os
import json
import numpy as np
import random

import glob
import librosa
import soundfile as sf

from tqdm import tqdm


def generate_noise_vec(root_path, type, sr=16000, is_train=True):
    noise_paths = []
    data_path = os.path.join(root_path, type)
    for name in glob.glob(data_path+"/*"):
        noise_paths.append(name)
    noise_list = []
    for path in noise_paths:
        noise, _ = librosa.load(path, sr=sr)
        print(noise.shape)
        noise_list.append(noise)
    noise_vec = np.concatenate(noise_list)
    # noise_vec = np.expand_dims(noise_vec, axis=-1)
    if is_train:
        sf.write('noise_train' + '.wav', noise_vec, sr)
    else:
        sf.write('noise_test' + '.wav', noise_vec, sr)


def get_path(clean_path, tra_num, val_num):
    path_list = []
    for path in glob.glob(clean_path+"/*.WAV"):
        path_list.append(path)
    train_index = np.random.randint(0, len(path_list), tra_num)
    val_index = np.random.randint(0, len(path_list), val_num)
    tra_paths = [path_list[index] for index in train_index]
    val_paths = [path_list[index] for index in val_index]
    with open("datasets/TIMIT/16k_DARCN/logs/tra_path_info.txt", 'w') as f:
        for path in tra_paths:
            f.write(path+"\n")
    with open("datasets/TIMIT/16k_DARCN/logs/val_path_info.txt", 'w') as f:
        for path in val_paths:
            f.write(path+"\n")
    # add test
    test_paths = []
    for path in glob.glob("datasets/TIMIT/16k/clean/test"+"/*.WAV"):
        test_paths.append(path)
    print(len(test_paths))
    with open("datasets/TIMIT/16k_DARCN/logs/test_path_info.txt", 'w') as f:
        for path in test_paths:
            f.write(path+"\n")


def mix_one(noisy_path, clean_path, noise_path, snr):
    clean_name = clean_path.split('/')[-1]
    speaker = clean_name.split('_')[0]
    noise_name = noise_path.split('/')[-1].split('.')[0]
    noisy_name = str(snr) + '+' + noise_name + '+' + speaker + '+' + clean_name
    # print(os.path.join(noisy_path, noisy_name))
    clean, clean_sr = librosa.load(clean_path, sr=16000)
    noise, noise_sr = librosa.load(noise_path, sr=16000)
    if len(noise) < len(clean) * 3:
        noise = np.tile(noise, int(np.ceil(len(clean) * 3 / len(noise))))
    start = random.randint(0, noise.shape[0] - clean.shape[0])
    noise_b = noise[start:start+clean.shape[0]]
    sum_clean = np.sum(clean ** 2)
    sum_noise = np.sum(noise_b ** 2)

    x = np.sqrt(sum_clean / (sum_noise * pow(10, snr / 10.0)))
    noise_c = x * noise_b
    noisy = clean + noise_c
    sf.write(os.path.join(noisy_path, noisy_name), noisy, clean_sr)


def mix_tra_eval_test():
    train_paths = []
    eval_paths = []
    test_paths = []
    with open('datasets/TIMIT/16k_DARCN/logs/tra_path_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_paths.append(line.strip())
        print(len(train_paths))
    with open('datasets/TIMIT/16k_DARCN/logs/val_path_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            eval_paths.append(line.strip())
        print(len(eval_paths))
    with open('datasets/TIMIT/16k_DARCN/logs/test_path_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_paths.append(line.strip())
        print(len(test_paths))
    # mix
    train_val_snrs = np.arange(-5, 16)
    test_snrs = [-5, 0, 5, 10]
    noisy_train = "datasets/TIMIT/16k_DARCN/noisy/train"
    noisy_eval = "datasets/TIMIT/16k_DARCN/noisy/eval"
    noisy_test = "datasets/TIMIT/16k_DARCN/noisy/test"
    for snr in tqdm(train_val_snrs):
        for path in train_paths:
            mix_one(noisy_train, path, "datasets/TIMIT/16k_DARCN/all_noise/noise_train.wav", snr)
    for snr in tqdm(train_val_snrs):
        for path in eval_paths:
            mix_one(noisy_eval, path, "datasets/TIMIT/16k_DARCN/all_noise/noise_train.wav", snr)
    for snr in tqdm(test_snrs):
        for path in test_paths:
            mix_one(noisy_test, path, "datasets/TIMIT/16k_DARCN/all_noise/noise_test.wav", snr)


mix_tra_eval_test()
