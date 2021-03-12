import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# data
import sidechainnet as scn
from sidechainnet.sequence.utils import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

# models
from clynmut.utils import *
from alphafold2_pytorch.utils import *


class MutPredict(torch.nn.Module):
    def __init__():
        return

    def forward(self, seqs, info=None, verbose=0):
        """ Predicts the mutation effect in a protein. 
            Inputs:
            * seqs: list of strings (batch, seqs). Sequence in 1-letter AA code.
            * info: any info required. 
            * verbose: int. verbosity level (0-silent, 1-minimal, 2-full)
        """
        seq_embedds = embedd_batch(seqs) # (batch, embedd_size)

        pred = None
        return pred

