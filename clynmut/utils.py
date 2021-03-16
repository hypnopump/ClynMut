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


#######################
##### MODEL UTILS #####
#######################

class Hier_Helper():
    def __init__(self, hier_graph):
        """ Helper class for Hierarchical classification. 
            Builds a DAG given a hierarchical graph. 
        """
        self.hier_graph = hier_graph
        self.nodes = self.build_graph(hier_graph)
        self.max_width = max([len(node["children"]) for node in self.nodes])
        # easy access
        self.idx2class = {node["idx"]: node["class"] for node in self.nodes}
        self.class2idx = {v:k for k,v in self.idx2class.items()} 
        # helpers for node access when deep pos:
        self.sample_pred = {} # TODO
        self.idx2route = [self.route2node(i) for i in range(len(self.nodes))]

    def build_graph(self, hier_graph):
        """ Builds the nodes iteratively in BFS fashion. """
        node_list = []
        frontier = [hier_graph]
        while len(frontier) > 0:
            node, frontier = frontier[0], frontier[1:]
            # build node
            node_attrs = {"parent_idx": None,
                          "idx": len(node_list),
                          "class": node["class"],
                          "children": [xi["class"] for xi in node["children"]],
                          "children_idxs": []}
            # special properties
            node_attrs["root"] = node_attrs["idx"] == 0
            node_attrs["terminal"] = len(node_attrs["children"]) == 0
            # add to tracker
            node_list.append(node_attrs)
        # updates nodes to include idxs of children and parent node
        for i, node in enumerate(node_list):
            if not node["root"]:
                children_idxs = []
                for j in range(len(node_list)):
                    # find parent node
                    if node["class"] == node_list[j]["children"]:
                        parent = j 
                    # find idxs of children
                    if node_list[j]["class"] in node["children"]:
                        children_idxs.append(j)
            # update nodes
            node_list[i]["parent_idx"] = partent
            node_list[i]["children_idxs"] = children_idxs

        return node_list 

    def dag(self, x, model_arch=[]):
        """ Follows the model architecture in a DAG fashion. 
            Inputs: 
            * x: inputs (batch, hidden_dims)
            * model_arch: model architecture in same order as self.nodes
            Output: (batch, nodes, hidden_dims)
        """
        device = x.device
        collect = []
        buffer_hidden = {}
        for i,node in enumerate(self.nodes):
            # get hidden state for that level and save for later preds
            hidden = model_arch[i]["hidden"](x)
            buffer_hidden[node["class"]] = hidden
            # classify
            clf = model_arch[i]["clf"](hidden)
            collect.append(clf)
            
        stacked = torch.stack(collect, dim=0).to(device)
        return rearrange(stacked, 'nodes b dims -> b nodes dims')

    def route2node(self, node_class=None, node_idx=None, return_format="class"):
        """ Finds the hierarchical route to a given node.
            Operate with idx, transform to class if not ready 
        """
        if node_idx == None:
            node_idx = self.class2idx[node_class]
        # start with node: 
        route = [node_idx]
        while route[-1] != 0:
            route.append(self.nodes[route[-1]]["parent_idx"])

        # back to class if required. default.
        if return_format == "class":
            return [self.idx2class[r] for r in route[::-1]]
        else:
            return route[::-1]

    def full2dict(self, batch):
        """ Converts a batch of preds into a list of dict preds.
            Inputs: 
            * batch: (bacth, nodes, hidden_dims)
            Outputs: list (length=batch) of pred_dicts
        """
        pred_list = []
        for example in batch:
            pred_dict = self.sample_pred.copy()
            for i, node in enumerate(example):
                # exploit -dict pointing instead of copying- for assigning tensors
                to_fill_dict = pred_dict
                # follow route
                for j,step in enumerate(self.idx2route[i][1:]):
                        to_fill_dict = to_fill_dict["children"][step]
                to_fill_dict["assign"] = node
            # add pred to batch list
            pred_list.append(pred_dict)

        return pred_list




########################
### POST-MODEL UTILS ###
########################

def hier_softmax(true_dict=None, pred_dict=None,
                 hier_graph=None, weight_mode="exp", 
                 criterions=None, verbose=0):
    """ Returns weighted softmax loss for hierarchical clf results. 
        Inputs:
        * true_dict: dict containing pairs of (classes, preds) for every level
        * pred_dict: dict containing pairs of (classes, preds) for every level
        * hier_graph: dict specifying relations between classes. not used for now
        * weight_mode: "div" for 1/(1+depth) or "exp" for 1/(2**depth)
        * criterions: list of torch.nn.CrossEntropyLoss instances
        * verbose: 0 for silent, 1 for full verbosity
    """
    loss = 0.
    depth_adj = lambda x: 1/(1+x)if weight_mode == "div" else 1/2**x 
    # select the first level, then go down hierarchical tree
    level = 0
    next_key = True
    level_pred_dict = pred_dict
    level_true_dict = true_dict
    level_hier_graph = hier_graph
    while next_key:
        loss_level = criterions[level_hier_graph["class"]](level_pred_dict["assign"],
                                                           level_true_dict["assign"])
        loss += loss_level.sum() * depth_adj(level)
        # log
        if verbose: 
            print("Level {0}, Parent: {1}, Children: {2}, Loss: {3}".format(
                  level, level_hier_graph["class"],
                  [xi["class"] for xi in level_hier_graph["children"]], 
                  loss)
            )
        # continue to next level if children or break
        next_key = False
        if len(level_true_dict["children"]) > 0:
            level += 1
            # select children label
            level_true_dict = level_true_dict["children"][0]
            # select dict of children / hier_graph which contains the same parent class
            level_pred_dict = [child for child in level_pred_dict["children"] \
                               if child["class"] == level_true_dict["class"]][0]
            level_hier_graph = [child for child in level_hier_graph["children"] \
                               if child["class"] == level_true_dict["class"]][0]
            next_key = True
        
    return loss




