
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import PrepData

DEVICE = torch.device("mps")

class Batch:
    def __init__(self, src, target, pad=0) -> None:
        # convert words id to long format.
        src = torch.from_numpy(src).to(device=DEVICE).long()
        tgt = torch.from_numpy(target).to(device=DEVICE).long()
        self.src = src

    @staticmethod
    def make_std_mask(tgt, pad):
        pass

class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    dataloader = PrepData(train_file='data/train.txt', eval_file='data/test.txt')
    en, cn = dataloader.load_data("data/train.txt")
    debug_k = 10
    # print(en[:debug_k])
    # print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
    # print(torch.backends.mps.is_built()) #MPS is activated