# Author: Eric Alcaide
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# models
from clynmut.utils import *
import alphafold2_pytorch.utils as af2utils
# import esm # after installing esm


# (e-)swish activation(s)
# https://arxiv.org/abs/1801.07145

class e_Swish_(torch.nn.Module):
    def forward(self, x, beta=1.1):
        return beta * x * x.sigmoid()

SiLU = e_Swish_


class Net_3d(torch.nn.Module):
    """ Gets an embedding from a 3d structure. 
        Not an autoencoder, just a specific encoder for
        this usecase.
        Will likely use GVP or E(n)-GNN:
        https://github.com/lucidrains/geometric-vector-perceptron/ 
    """
    def __init__(self):
        return

    def forward(self, coords, cloud_mask):
        """ Gets an embedding from a 3d structure. """
        return


class Hier_CLF(torch.nn.Module):
    """ Hierarchical classification/regression module. """
    def __init__(self, hier_graph={}, hidden_dim=None):
        self.hier_graph = hier_graph
        self.hier_scaff = Hier_Helper(hier_graph)
        self.hidden_dim = hidden_dim
        self.arch = []
        # build node MLPs
        for i,node in enumerate(self.hier_scaff.nodes):
            dims_in = self.hier_scaf.max_width if i!=0 else self.hidden_dim
            dims_out = self.hier_scaf.max_width
            self.arch.append({"class": node["class"],
                              "hidden": torch.nn.Sequential(
                                            torch.nn.Linear(dims_in,
                                                            dims_out),
                                            SiLU(),
                                        ),
                              "clf": torch.nn.Sequential(
                                         torch.nn.Linear(dims_out,
                                                         dims_out)
                                         )
                             })

    def forward(self, x, pred_format="dict"):
        """ The custom architecture for a hierarchical classification.
            Defines the MLPs and final gaussian processes for each node.
            Inputs: 
            * x: (batch, hidden) tensor
            * pred_format: one of ["dict", "tensor"]
        """
        full_pred = self.hier_scaff.dag(x, self.arch)
        if pred_format == "dict":
            pred_dict = self.hier_scaff.full2dict(full_pred)
        return full_pred


class MutPredict(torch.nn.Module):
    def __init__(self,
                 seq_embedd_dim = 1280, # 
                 struct_embedd_dim = 256, 
                 seq_reason_dim = 128, 
                 struct_reason_dim = 128,
                 hier_graph = {},
                 dropout = 0.0,
                 use_msa = False,
                 msa_max_seq = 256, # max number of MSA sequences to read.
                 device = None):
        """ Predicts the phenotypical impact of mutations. """
        self.seq_embedd_dim = seq_embedd_dim
        self.seq_reason_dim = seq_reason_dim
        self.struct_embedd_dim = struct_embedd_dim
        self.struct_reason_dim = struct_reason_dim
        # take same value for the 3 parts if no list is passed.
        self.dropout = [dropout]*3 if isinstance(dropout, float) else dropout
        self.hier_graph = hier_graph

        # nlp arch - no gradients here
        self.use_msa = use_msa
        self.msa_max_seq = msa_max_seq
        if use_msa:
            ##  alternatively do
            # import esm # after installing esm
            # embedd_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            embedd_model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
            batch_converter = alphabet.get_batch_converter()
        else:
            ##  alternatively do
            # embedd_model, alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
            embedd_model, alphabet = torch.hub.load("facebookresearch/esm", "esm_msa1_t12_100M_UR50S") 
            batch_converter = alphabet.get_batch_converter()

        self.nlp_stuff = [embedd_model, alphabet, batch_converter]
        self.seq_embedder = partial(embedd_seq_batch,
                                    embedd_model=embedd_model, batch_converter=batch_converter)

        # 3d module
        self.struct_embedder = Net_3d()
        # reasoning modules
        self.nlp_mlp = torch.nn.Sequential(
                              torch.nn.Linear(2*seq_embedd_dim,
                                              seq_reason_dim*2),
                              torch.nn.Dropout(self.dropout[0]),
                              SiLU(),
                              torch.nn.Linear(seq_reason_dim * 2, 
                                              seq_reason_dim),
                              torch.nn.Dropout(self.dropout[0]),
                              SiLU(),
                          )
        self.struct_mlp = torch.nn.Sequential(
                              torch.nn.Linear(struct_reason_dim,
                                        struct_reason_dim*2),
                              torch.nn.Dropout(self.dropout[1]),
                              SiLU(),
                              torch.nn.Linear(struct_reason_dim * 2, 
                                              struct_reason_dim),
                              torch.nn.Dropout(self.dropout[1]),
                              SiLU(),
                          )
        self.common_mlp = torch.nn.Sequential(
                              torch.nn.Linear(struct_reason_dim + seq_reason_dim,
                                        struct_reason_dim + seq_reason_dim),
                              torch.nn.Dropout(self.dropout[-1]),
                              SiLU()
                          )
        # classifier
        self.hier_clf = Hier_CLF(hier_graph, hidden_dim=struct_reason_dim+seq_reason_dim)
        return

    def forward(self, seqs, msa_routes=None, coords=None, cloud_mask=None,
                pred_format="dict", info=None, verbose=0):
        """ Predicts the mutation effect in a protein. 
            Inputs:
            * seqs: (2, b) list of pairs (wt and mut) of strings.
                    Sequences in 1-letter AA code.
            * msas: (2, b) list of pairs (wt and mut) of routes to msa files .
            * coords: (b, l, c, 3) coords array in sidechainnet format
            * cloud_mask: (b, l, c) boolean mask on actual points from coords
            * pred_format: one of ["dict", "tensor"]
            * info: any info required. 
            * verbose: int. verbosity level (0-silent, 1-minimal, 2-full)
        """
        scaffold = torch.zeros(len(seqs), self.seq_reason_dim+self.struct_reason_dim) 

        # NLP
        # MSATransformer if possible
        if msas is not None:
            wt_seq_data = [ af2utils.read_msa( filename=msa_route, nseq=self.msa_max_seq ) \
                            for msa_route in msa_routes[0]]
            mut_seq_data = [ af2utils.read_msa( filename=msa_route, nseq=self.msa_max_seq ) \
                             for msa_route in msa_routes[1]]
        else: 
            wt_seq_data, mut_seq_data = None, None

        wt_seq_embedds = self.seq_embedder(seqs[0], wt_seq_data) # (batch, embedd_size)
        mut_seq_embedds = self.seq_embedder(seqs[1], mut_seq_data) # (batch, embedd_size)

        # reason the embedding
        seq_embedds = torch.cat([wt_seq_embedds, mut_seq_embedds], dim=-1)
        scaffold[:, :-self.struct_reason_dim] = self.nlp_mlp(seq_embedds)

        # 3D
        # only do if passed
        if coords is not None and cloud_mask is not None:
            struct_embedds = self.struct_embedder(coords, cloud_mask)
            scaffold[:, -self.struct_reason_dim:] = self.struct_mlp(struct_embedds)

        # common step
        x = self.common_mlp(scaffold)
        return self.hier_clf(x, pred_format=pred_format)

    def __repr__(self):
        return "ClynMut model with following args: "+str(self.__dict__)




