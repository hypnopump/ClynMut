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
    criterion = torch.nn.CrossEntropyLoss()
    true_dict = {"class": "all",
                 "classes": ["class_1", "class_2", "class_3"], 
                 "assign": torch.tensor([1]),
                 "children": [
                  {
                   "class": "class_2",
                   "classes": ["class_21", "class_22", "class_23"], 
                   "assign": torch.tensor([2]),
                   "children" : {}
                  },]
                }
    pred_dict = {"class": "all", 
                 "assign": torch.tensor([[0.13, 0.53, 0.34]]),
                 "children": [
                  {
                   "class": "class_2",
                   "assign": torch.tensor([[0.25, 0.2, 0.55]]),
                   "children" : {}
                  },
                  {
                   "class": "class_1",
                   "assign": torch.tensor([[0., 0., 1.]]),
                   "children" : {}
                  },]
                }

    hier_graph = {"class": "all",
                  "classes": ["class_1", "class_2", "class_3"], 
                  "children": [
                   {
                    "class": "class_1",
                    "classes": ["class_11", "class_12", "class_13"], 
                    "children": [
                     {"class": "class_11", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_12", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_13", 
                      "classes": [],
                      "children": []
                     },]
                   },
                   {
                   	"class": "class_2",
                    "classes": ["class_21", "class_22", "class_23"], 
                    "children": [
                     {"class": "class_21", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_22", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_23", 
                      "classes": [],
                      "children": []
                     },]
                   },
                   {
                   	"class": "class_3",
                    "classes": ["class_31", "class_32", "class_33"], 
                    "children": [
                     {"class": "class_31", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_32", 
                      "classes": [],
                      "children": []
                     },
                     {"class": "class_33", 
                      "classes": [],
                      "children": []
                     },]
                   },
                  ]}

    res = hier_softmax(true_dict, pred_dict, hier_graph=hier_graph,
    										 weight_mode=None,
    										 criterion=criterion)
    assert list(res.shape) == []

if __name__ == "__main__":
	test_hier_softmax()




