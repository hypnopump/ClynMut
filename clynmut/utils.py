import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# data
import sidechainnet as scn
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

# models
# from alphafold2_pytorch.utils import *


# Constants / Config
PADDING_TOKEN = "_"
FEATURES = None # one of ["esm", "msa", None]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
SAVE_DIR = ""


#######################
### PRE-MODEL UTILS ###
#######################


# set emebdder model from esm if appropiate - Load ESM-1b model
if FEATURES == "esm":
    # from pytorch hub (almost 30gb)
    EMBEDD_MODEL, ALPHABET = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
    ##  alternatively do
    # import esm # after installing esm
    # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    BATCH_CONVERTER = alphabet.get_batch_converter()


def embedd_seq_batch(seqs):
    """ Embedds a batch of sequences into mean of all AA embeddings.
        Inputs:
        * seqs: iterator of strs (batch, seqs)
        Outputs: float tensor of size (batch, embedd_dims)
    """
    embedd_list = []
    for seq in seqs:
        seq_pure = seq.rstrip(PADDING_TOKEN)
        embedd = get_esm_embedd(seq_pure)
        # take mean
        embedd_list.append( embedd.mean(dim=0, keepdim=True) )

    return torch.cat(embedd_list, dim=0).to(DEVICE)


def get_esm_embedd(seq):
    """ Returns embeddings for all AAs of a protein sequence.
        Inputs: 
        * seq: str or tensor of ints (sidechainnet key-pair table) of length N
        Outputs: float tensor of size (N, D)
    """
    # handles both sequences and int encodings
    if isinstance(seq, torch.Tensor):
        seq = "".join([VOCAB._int2char[x]for x in seq.cpu().numpy()])
    # adjust input data and embedd
    batch_labels, batch_strs, batch_tokens = batch_converter( [(0, seq)] )
    with torch.no_grad():
        results = embedd_model(batch_tokens, repr_layers=[33], return_contacts=False)
    return results["representations"][33].to(DEVICE)


########################
### POST-MODEL UTILS ###
########################

def hier_softmax(true_dict, pred_dict, hier_graph=None, weight_mode=None, criterion=None):
    """ Returns weighted softmax loss for hierarchical clf results. 
        Inputs:
        * true_dict: dict containing pairs of (classes, preds) for every level
        * pred_dict: dict containing pairs of (classes, preds) for every level
        * hier_graph: dict specifying relations between classes. not used for now
        * weight_mode: defaults to 1/(1+depth)
        * loss: torch.nn.CrossEntropyLoss instance
    """
    loss = 0.
    # select the first level, then go down hierarchical tree
    level = 0
    next_key = True
    level_pred_dict = pred_dict
    level_true_dict = true_dict
    while next_key:
        loss_level =  criterion(level_pred_dict["assign"],
                                level_true_dict["assign"])
        loss += loss_level.sum() / (1+level)
        # select dict of children which contains the same parent class
        next_key = False
        for child in level_pred_dict["children"]:
        	if child["class"] == level_true_dict["class"]:
        		level_pred_dict = child
        		next_key = True
        		break
        # select target
        level += 1
        level_true_dict = level_true_dict["children"][0]
    return loss




