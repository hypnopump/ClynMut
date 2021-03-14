# Author: Eric Alcaide
import os
import sys
sys.path.append("../clynmut")

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# data
import sidechainnet as scn
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

# models
from clynmut.utils import *
# from alphafold2_pytorch.utils import *


class Net_3d(torch.nn.Module):
    """ Gets an embedding from a 3d structure. 
        Not an autoencoder, just a specific encoder for
        this usecase.
        Will likely use GVP:
        https://github.com/lucidrains/geometric-vector-perceptron/ 
    """
    def __init__(self):
        return

    def forward(self, coords, cloud_mask):
        """ Gets an embedding from a 3d structure. """
        return


class Hier_CLF(torch.nn.Module):
    """ Hierarchical classification module. """
    def __init__(self, hier_graph={}):
        self.hier_graph = hier_graph
        self.hier_scaff = Hier_Helper(hier_graph)
        self.arch = []

    def forward(self, x):
        """ The custom architecture for a hierarchical classification.
            Defines the MLPs and final gaussian processes for each node.
        """
        full_pred = self.hier_scaff.dag(x, self.arch)
        pred_dict = self.hier_scaff.full2dict(full_pred)
        return pred_dict


class MutPredict(torch.nn.Module):
    def __init__(self,
                 seq_embedd_dim = 512,
                 struct_embedd_dim = 256, 
                 seq_reason_dim = 512, 
                 struct_reason_dim = 256,
                 hier_graph = {},
                 dropout = 0.0,
                 use_msa = False,
                 device = None):
        """ Predicts the phenotypical impact of mutations. """
        self.seq_embedd_dim = seq_embedd_dim
        self.seq_reason_dim = seq_reason_dim
        self.struct_embedd_dim = struct_embedd_dim
        self.struct_reason_dim = struct_reason_dim
        # take same value for the 3 parts if no list is passed.
        self.dropout = [dropout]*3 if isinstance(dropout, float) else dropout
        self.hier_graph = hier_graph

        # nlp arch
        self.use_msa = use_msa
        if use_msa:
            self.msa_embedder = embedd_msa_batch
        else:
            self.seq_embedder = embedd_seq_batch
        # 3d module
        self.struct_embedder = Net_3d()
        # reasoning modules
        self.nlp_mlp = None
        self.struct_mlp = None
        self.common_mlp = None
        # classifier
        self.hier_clf = Hier_CLF(hier_graph)
        return

    def forward(self, seqs, msas=None, coords=None, cloud_mask=None,
                info=None, verbose=0):
        """ Predicts the mutation effect in a protein. 
            Inputs:
            * seqs: (b,) list of strings. Sequence in 1-letter AA code.
            * msas: (b,) list of outes to msa files .
            * coords: (b, l, c, 3) coords array in sidechainnet format
            * info: any info required. 
            * verbose: int. verbosity level (0-silent, 1-minimal, 2-full)
        """
        scaffold = torch.zeros(len(seqs), self.seq_reason_dim+self.struct_reason_dim) 

        # NLP
        # MSATransformer if possible
        if msas is not None:
            seq_embedds = self.msa_embedder(msas)
        # ESM1b if no MSA
        else:
            seq_embedds = self.seq_embedder(seqs) # (batch, embedd_size)
        # reason the embedding
        scaffold[:, :-self.struct_reason_dim] = self.nlp_mlp(seq_embedds)

        # 3D
        # only do if passed
        if coords is not None and cloud_mask is not None:
            struct_embedds = self.struct_embedder(coords, cloud_mask)
            scaffold[:, -self.struct_reason_dim:] = self.struct_mlp(struct_embedds)

        # common step
        x = self.common_mlp(scaffold)
        return self.hier_clf(x)

    def __repr__(self):
        return "ClynMut model with following args: "+str(self.__dict__)




