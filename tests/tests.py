# Author: Eric Alcaide
import os
import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# data
import sidechainnet as scn

# models
from clynmut.utils import *

def test_hier_softmax():
    true_dict = {"class": "all",
                 "assign": torch.tensor([1]),
                 "children": [
                  {
                   "class": "class_2",
                   "assign": torch.tensor([2]),
                   "children" : []
                  },]
                }
    pred_dict = {"class": "all", 
                 "assign": torch.tensor([[0.13, 0.53, 0.34]]),
                 "children": [
                  {
                   "class": "class_2",
                   "assign": torch.tensor([[0.25, 0.2, 0.55]]),
                   "children" : []
                  },
                  {
                   "class": "class_1",
                   "assign": torch.tensor([[0., 0., 1.]]),
                   "children" : []
                  },]
                }

    hier_graph = {"class": "all",
                  "children": [
                   {
                    "class": "class_1",
                    "children": [
                     {"class": "class_11", 
                      "children": []
                     },
                     {"class": "class_12", 
                      "children": []
                     },
                     {"class": "class_13", 
                      "children": []
                     },]
                   },
                   {
                   	"class": "class_2",
                    "children": [
                     {"class": "class_21", 
                      "children": []
                     },
                     {"class": "class_22", 
                      "children": []
                     },
                     {"class": "class_23", 
                      "children": []
                     },]
                   },
                   {
                   	"class": "class_3",
                    "children": [
                     {"class": "class_31", 
                      "children": []
                     },
                     {"class": "class_32", 
                      "children": []
                     },
                     {"class": "class_33", 
                      "children": []
                     },]
                   },
                  ]}
    nodes_list = ["all", "class_1", "class_2", "class_3"]
    criterions = {node: torch.nn.CrossEntropyLoss() for node in nodes_list}

    res = hier_softmax(true_dict, pred_dict, hier_graph=hier_graph,
    										 weight_mode=None,
    										 criterions=criterions,
    										 verbose=1)
    assert list(res.shape) == []

if __name__ == "__main__":
	test_hier_softmax()




