import os
import sys
sys.path.append('/home/judiebig/code/twx')

import json
import argparse
import time
from tools.utils import *
from torch.utils.data import DataLoader
import numpy as np
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


"""
训练并保存模型
"""
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("-C", "--config", required=True, type=str,
                    help="Specify the configuration file for training (*.json).")
args = parser.parse_args()
# setup_seed(2020)

torch.cuda.set_device(0)


def main(opt):
    model = initialize_config(opt['model'])
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    start_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        filename=opt['log_path']+model.__class__.__name__+".txt",
                        filemode='w')
    m_print(f"start logging time:\t{start_time}")
    m_print(json.dumps(opt, indent=4))

    train_dataset = initialize_config(opt['train_dataset'])
    print(train_dataset.__len__())
    train_dataloader = DataLoader(
        shuffle=config["train_dataloader"]["shuffle"],
        dataset=train_dataset,   # 生成单条数据
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        collate_fn=pad_to_longest,
        drop_last=True
    )

    eval_dataset = initialize_config(opt['validation_dataset'])
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"],
        collate_fn=pad_to_longest,
        shuffle=config["validation_dataloader"]["shuffle"]
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], 0.999)
    )

    loss_function = initialize_config(config['loss_function'])

    trainer = initialize_config(config['trainer'])
    trainer.initialize(config, model, optimizer, loss_function, train_dataloader, eval_dataloader)
    # trainer.train(opt)
    trainer.train()

if __name__ == '__main__':
    config = json.load(open(args.config))
    main(config)
