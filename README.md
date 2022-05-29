# IDANet

![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/pytorch-1.6-green)
![](https://img.shields.io/badge/cudatoolkit-10.1-green)
![](https://img.shields.io/badge/cuda-11.0-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)
![](https://img.shields.io/badge/pystoi-0.3.3-green)
![](https://img.shields.io/badge/pypesq-1.2.4-green)

This repo provides a reference implementation of **IDANet** as described in the paper:

> IDANet: An Information Distillation and Aggregation Network for Speech Enhancement

> Accepted by SPL 2021

## data preprosessing

105 types of noise are concatenated for training and validation, while 110 types of noise for testing (add 5 unseen noises).  The ratio of the invisible part to the visible part of our test noise is about 4:1 (5 unseen noises vs. 105 seen noises, 20 minutes vs. 5 minutes)

The model's code we use for DARCN is from their official depository (https://github.com/Andong-Li-speech/DARCN). For CRN, we use an unofficial code implemented from https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement. Since GRN did not publish their code, we reproduce it according to the original paper. 

The  experimental platform  is  Ubuntu  LTS  18.04  with  i7-9700  and  RTX  2060.


## References
```bib
If you find the code useful for your research, please consider citing
@article{tai2021idanet,
  title={IDANet: An Information Distillation and Aggregation Network for Speech Enhancement},
  author={Tai, Wenxin and Lan, Tian and Wang, Qianhui and Liu, Qiao},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1998--2002},
  year={2021},
  publisher={IEEE}
}
```

## Contact

For any questions please open an issue or drop an email to: `wxtai@std.uestc.edu.cn`
