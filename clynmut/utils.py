import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# data
import sidechainnet as scn
from sidechainnet.utils.sequence import VOCAB
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES


# Constants / Config
PADDING_TOKEN = "_"
FEATURES = None # one of ["esm", "msa", None]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
SAVE_DIR = ""


#######################
### PRE-MODEL UTILS ###
#######################

def get_esm_embedd(seq, embedd_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein. 
        Inputs: 
        * seq: (L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        * embedd_type: one of ["mean", "per_tok"]. 
                       "per_tok" is recommended if working with sequences.
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for ESM-1b
            * embedd_dim: number of embedding dimensions. 
                          768 for MSA_Transformer and 1280 for ESM-1b
    """
    str_seq = "".join([VOCAB._int2char[x] for x in seq.cpu().numpy()])
    # use MSA transformer
    if msa_data is not None: 
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = batch_converter(msa_data)
        with torch.no_grad():
            results = embedd_model(msa_batch_tokens.to(seq.device), repr_layers=[12], return_contacts=False)
        # index 0 is for start token. so take from 1 one
        token_reps = results["representations"][12][0, :,  1 : len(str_seq) + 1]
        
    # base ESM case
    else: 
        batch_labels, batch_strs, batch_tokens = batch_converter( [(0, str_seq)] )
        with torch.no_grad():
            results = embedd_model(batch_tokens.to(seq.device), repr_layers=[33], return_contacts=False)
        # index 0 is for start token. so take from 1 one
        token_reps = results["representations"][33][:, 1 : len(str_seq) + 1].unsqueeze(dim=1)
    
    return token_reps.mean(dim=-2) if embedd_type == "mean" else token_reps


def get_t5_embedd(seq, tokenizer, encoder, msa_data=None, device=None, embedd_type="per_tok"):
    """ Returns the ProtT5-XL-U50 embeddings for a protein.
        Supports batched embedding as well as individual.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * tokenizer:  tokenizer model: T5Tokenizer
        * model: encoder model: T5EncoderModel
                 ex: from transformers import T5EncoderModel, T5Tokenizer
                     model_name = "Rostlab/prot_t5_xl_uniref50"
                     tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
                     model = T5EncoderModel.from_pretrained(model_name)
                     # prepare model 
                     model = model.to(device)
                     model = model.eval()
                     if torch.cuda.is_available():
                         model = model.half()
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for T5 models
            * embedd_dim: number of embedding dimensions. 1024 for T5 models
    """
    # get params and prepare
    device = seq.device if device is None else device
    # format accordingly
    embedd_inputs = []
    seq = seq.unsqueeze(0) if len(seq.shape[0]) < 2 else seq
    for ids in seq.cpu().tolist():
        chars = ' '.join([id2aa[i] for i in ids])
        chars = re.sub(r"[UZOB]", "X", chars)
        out.append(chars)
    
    # embedd - https://huggingface.co/Rostlab/prot_t5_xl_uniref50
    inputs_embedding = []
    shift_left, shift_right = 0, -1
    ids = tokenizer.batch_encode_plus(embedd_inputs, add_special_tokens=True,
                                                     padding=True, 
                                                     return_tensors="pt")
    with torch.no_grad():
        embedding = model(input_ids=torch.tensor(ids['input_ids']).to(device), 
                          attention_mask=torch.tensor(ids["attention_mask"]).to(device),
                          decoder_input_ids=None)
    # return (batch, n_seqs, seq_len, embedd_dim)
    token_reps = embedding.last_hidden_state[:, shift_left:shift_right].to(device).unsqueeze(1)

    return token_reps.mean(dim=-2) if embedd_type == "mean" else token_reps


def embedd_seq_batch_esm(seqs, msa_data=None, embedd_model=None, batch_converter=None):
    """ Embedds a batch of sequences get_esm_embeddinto mean of all AA embeddings.
        Inputs:
        * seqs: iterator of strs (batch, seqs)
        Outputs: float tensor of size (batch, embedd_dims)
    """
    embedd_list = []
    for i,seq in enumerate(seqs):
        seq_pure = seq.rstrip(PADDING_TOKEN)
        msa_pure = [msa_data[i]] if msa_data is not None else None
        # take mean
        embedd_list.append( get_esm_embedd(seq_pure, msa_data=msa_pure,
                                                     embedd_model=EMBEDD_MODEL, 
                                                     batch_converter=BATCH_CONVERTER, 
                                                     embedd_type="mean"))
    return torch.stack(embedd_list, dim=-2).to(DEVICE)


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
            node_list[i]["parent_idx"] = parent
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
    depth_adj = lambda x: 1/(1+x) if weight_mode == "div" else 1/2**x 
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




