from tools.utils import *
from tools.metrics import *
from tqdm import tqdm
import numpy as np
import pystoi
import pypesq
import os
import torch
import time
import gc

class TrainMag(object):
    def __init__(self):
        self.config = None
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.win_len = None
        self.hop_len = None
        self.n_fft = None
        self.best_score = 0
        self.save_root = None
        self.checkpoints_dir = None
        self.scheduler = None
        self.stft = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, config, model, optimizer, loss_function, train_dataloader, eval_dataloader):
        self.config = config
        self.model = model.cuda()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.win_len = config['stft_parameter']['win_len']
        self.hop_len = config['stft_parameter']['hop_len']
        self.n_fft = self.win_len

        self.save_root = config['save_path'] + model.__class__.__name__
        self.checkpoints_dir = os.path.join(self.save_root, 'checkpoints')
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)


    def _save_checkpoints(self, epoch, is_best=False):
        """
        Save checkpoints, if is_best, save as best model , or latest model, whatever save current epoch
        :param epoch:
        :param is_best:
        :return:
        """
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.cpu().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(self.checkpoints_dir, "latest_model.tar"))
        torch.save(state_dict["model"], os.path.join(self.checkpoints_dir, f"model_{str(epoch).zfill(4)}.pth"))
        if is_best:
            print('Found best score in {epoch} epoch'.format(epoch=epoch))
            torch.save(state_dict, os.path.join(self.checkpoints_dir, "best_model.tar"))
        self.model.cuda()

    @staticmethod
    def _calculate_score(**args):
        stoi = 0
        pesq = 0
        if 'stoi' in args.keys():
            stoi = args['stoi']
        if 'pesq' in args.keys():
            pesq = args['pesq']
        score = stoi + (pesq + 0.5) * 0.2
        return score

    def _is_best_score(self, score):
        """Check if the current model is the best model"""
        if score >= self.best_score:
            self.best_score = score
            return True
        else:
            # self.best_score = score
            return False

    def train(self):
        m_print(f"The amount of parameters in the project is {print_networks([self.model]) / 1e6} million.")

        for epoch in range(self.config['epochs']):
            total_loss = 0
            self.model.train()
            '''training'''
            for mixs_wav, cleans_wav, lengths, _ in tqdm(self.train_dataloader):

                self.optimizer.zero_grad()

                mixs = torch.stft(mixs_wav,
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_len,
                                  win_length=self.win_len,
                                  window=torch.hamming_window(self.win_len)).permute(0, 2, 1, 3).cuda()

                # mixs = self.stft.transform(mixs_wav.cuda())
                mixs_real = mixs[:, :, :, 0]
                mixs_imag = mixs[:, :, :, 1]
                mixs_mag = torch.sqrt(mixs_real ** 2 + mixs_imag ** 2)


                cleans = torch.stft(cleans_wav,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_len,
                                    win_length=self.win_len,
                                    window=torch.hamming_window(self.win_len)).permute(0, 2, 1, 3).cuda()

                cleans_real = cleans[:, :, :, 0]
                cleans_imag = cleans[:, :, :, 1]
                cleans_mag = torch.sqrt(cleans_real ** 2 + cleans_imag ** 2)

                # z_score
                # mixs_mag, _, _ = z_score(mixs_mag)
                # cleans_mag, _, _ = z_score(cleans_mag)

                enhances_mag = self.model(mixs_mag)

                frames = []
                for length in lengths:
                    frame = (length - self.win_len) // self.hop_len + 3
                    frames.append(frame)
                loss = self.loss_function.calculate_loss(enhances_mag, cleans_mag, frames)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                # break
                # print(loss.item())
                gc.collect()

            # break
            tqdm.write(f"\nepoch: {epoch}, total loss: {total_loss}")
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            m_print(f"epoch: {epoch} end logging time:\t{end_time}")

            self.model.eval()
            '''validating'''
            with torch.no_grad():
                stoi_sum = 0
                pesq_sum = 0
                si_sdr_sum = 0
                count_number = 0
                for mixs_wav, cleans_wav, lengths, _ in tqdm(self.eval_dataloader):
                    mixs = torch.stft(mixs_wav,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_len,
                                      win_length=self.win_len,
                                      window=torch.hamming_window(self.win_len)).permute(0, 2, 1, 3).cuda()
                    # mixs = self.stft.transform(mixs_wav.cuda())
                    mixs_real = mixs[:, :, :, 0]
                    mixs_imag = mixs[:, :, :, 1]
                    mixs_mag = torch.sqrt(mixs_real ** 2 + mixs_imag ** 2)

                    # z_score
                    # mixs_mag, mixture_mean, mixture_std = z_score(mixs_mag)

                    enhances_mag = self.model(mixs_mag)

                    # z_score
                    # enhances_mag = reverse_z_score(enhances_mag, mixture_mean, mixture_std)

                    '''eval'''
                    enhances_real = enhances_mag * mixs_real / mixs_mag
                    enhances_imag = enhances_mag * mixs_imag / mixs_mag
                    enhances = torch.stack([enhances_real, enhances_imag], 3)
                    enhances = enhances.permute(0, 2, 1, 3)

                    enhances_wav = torch.istft(enhances,
                                               n_fft=self.n_fft,
                                               hop_length=self.hop_len,
                                               win_length=self.win_len,
                                               window=torch.hamming_window(self.win_len).cuda(),
                                               length=max(lengths))
                    # enhances_wav = self.stft.inverse(enhances)

                    frames = []
                    # len_list = []
                    for length in lengths:
                        frame = (length - self.win_len) // self.hop_len + 3
                        frames.append(frame)
                        # len_list.append((frame - 1) * 160 + 320)

                    cleans_wav = cleans_wav.cpu().numpy()
                    enhances_wav = enhances_wav.cpu().numpy()
                    for clean, enhance, length in zip(cleans_wav, enhances_wav, lengths):
                        clean = clean[:length]
                        enhance = enhance[:length]
                        stoi_transform = pystoi.stoi(clean, enhance, 16000)
                        pesq_transform = pypesq.pesq(clean, enhance, 16000)
                        si_sdr_transform = SI_SDR(clean, enhance)
                        if np.isnan(stoi_transform) or np.isnan(pesq_transform) or np.isnan(si_sdr_transform):
                            continue
                        stoi_sum += stoi_transform
                        pesq_sum += pesq_transform
                        si_sdr_sum += si_sdr_transform
                        count_number += 1

                score = self._calculate_score(stoi=stoi_sum, pesq=pesq_sum)
                if self._is_best_score(score):
                    is_best = True
                else:
                    is_best = False
                self._save_checkpoints(epoch, is_best)

                print(count_number)
                m_print(f"stoi score: "
                        f"{stoi_sum / count_number},"
                        f"pesq score: "
                        f"{pesq_sum / count_number},"
                        f"si_sdr score: "
                        f"{si_sdr_sum / count_number},"
                        f"is best: {is_best}"
                        )
