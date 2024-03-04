
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device("mps")

def subsequent_mask(size: int) -> torch.FloatTensor:
    """_summary_

    Args:
        size (int): _description_

    Returns:
        torch.FloatTensor: _description_
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0 #shape: 1 x size x size 


class Batch:
    def __init__(self, src, target, pad=0) -> None:
        # convert words id to long format.
        src = torch.from_numpy(src).to(device=DEVICE).long() # shape: B x sequence_len
        tgt = torch.from_numpy(target).to(device=DEVICE).long() # shape: B x sequence_len
        self.src = src
        # get the padding postion binary mask. Change the matrix shape to Bx1×seq.length
        self.src_mask = (src != pad).unsqueeze(-2)
        # if target is not empty, mask decoder target.
        if target is not None:
            pass

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
    # en, cn = dataloader.load_data("data/train.txt")
    debug_k = 10
    print(subsequent_mask(10))
    # print(en[:debug_k])
    # print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
    # print(torch.backends.mps.is_built()) #MPS is activated