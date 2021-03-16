```
Sequence -----------+--> 3D_structure --> 3D_module --+                                      +--> ?
|                   |                                 |                                      +--> ?
|                   |                                 +--> Joint_module --> Hierarchical_CLF +--> ?
|                   |                                 |                                      +--> ?
+-> NLP_embeddings -+-------> Embedding_module -------+                                      +--> ?
```

## ClynMut: Predicting the Clynical Relevance of Genome Mutations (wip)

To be a next-generation DL-based phenotype prediction from genome mutations. Will use sota NLP and structural techniques. 


Planned modules will likely be: 
* 3D learning module 
* NLP embeddings
* Joint module
* Hierarchical classification 

The main idea is for the model to learn the prediction in an end-to-end fashion. 

## Example Usage: 

```python
import torch
from clynmut import *

hier_graph = {"class": "all", 
              "children": [
                {"class": "effect_1", "children": [
                  {"class": "effect_12", "children": []},
                  {"class": "effect_13", "children": []}
                ]},
                {"class": "effect_2", "children": []},
                {"class": "effect_3", "children": []},
              ]}

model = MutPredict(
    seq_embedd_dim = 512,
    struct_embedd_dim = 256, 
    seq_reason_dim = 512, 
    struct_reason_dim = 256,
    hier_graph = hier_graph,
    dropout = 0.0,
    use_msa = False,
    device = None)

seqs = ["AFTQRWHDLKEIMNIDALTWER",
        "GHITSMNWILWVYGFLE"]

pred_dicts = model(seqs, pred_format="dict")
```


## Important topics: 
### 3D structure learning

There are a couple architectures that can be used here. I've been working on 2 of them, which are likely to be used here: 
* <a href="https://github.com/lucidrains/geometric-vector-perceptron">GVP</a>
* <a href="https://github.com/lucidrains/egnn-pytorch">E(n)-gnn</a> 

### Hierarchical classification

* [x] A simple custom helper class has been developed for it.

## Testing

```bash
$ python setup.py test
```

## Datasets: 

This package will use the awesome work by <a href="http://github.com/jonathanking">Jonathan King</a> at <a href="https://github.com/jonathanking/sidechainnet">this repository</a>.

To install

```bash
$ pip install git+https://github.com/jonathanking/sidechainnet.git
```
Or

```bash
$ git clone https://github.com/jonathanking/sidechainnet.git
$ cd sidechainnet && pip install -e .
```

+++

* referenced in <a href="https://ieeexplore.ieee.org/document/9175781/">NLP-SNPPred</a>: 
    * training:
        * OncoKB : ()
    * validation: 
        * VariBench (three positive/pathogenic datasets: TP53, ClinVar and DoCM)
        * CIViC (expert-crowd-sourced knowledge base of variants in cancer. We only considered those examples that were labeled as "Likely Pathogenic”, “Poor Outcome” or “Negative”)

* referenced in <a href=https://pdfs.semanticscholar.org/b1c4/31717cf470634bfb5faca0c0ec9d3bd5ec66.pdf>MutPred</a>:
    * training:
        * Human Gene Mutation Database (HGMD)
        * Genome Aggregation Database (gnomAD)
    * validation:
        * COSMIC : (Catalogue Of Somatic Mutations In Cancer)
        * dbCID (DataBase of Cancer Driver InDel) : 
        * affected by autism pectrum disorder (ASD) from the REACH Project and the Simons Simplex Collection


## Citations:

```bibtex
@article{pejaver_urresti_lugo-martinez_pagel_lin_nam_mort_cooper_sebat_iakoucheva et al._2020,
    title={Inferring the molecular and phenotypic impact of amino acid variants with MutPred2},
    volume={11},
    DOI={10.1038/s41467-020-19669-x},
    number={1},
    journal={Nature Communications},
    author={Pejaver, Vikas and Urresti, Jorge and Lugo-Martinez, Jose and Pagel, Kymberleigh A. and Lin, Guan Ning and Nam, Hyun-Jun and Mort, Matthew and Cooper, David N. and Sebat, Jonathan and Iakoucheva, Lilia M. et al.},
    year={2020}
```

```bibtex
@article{rehmat_farooq_kumar_ul hussain_naveed_2020, 
    title={Predicting the pathogenicity of protein coding mutations using Natural Language Processing},
    DOI={10.1109/embc44109.2020.9175781},
    journal={2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)},
    author={Rehmat, Naeem and Farooq, Hammad and Kumar, Sanjay and ul Hussain, Sibt and Naveed, Hammad},
    year={2020}
```

```bibtex
@article{pagel_antaki_lian_mort_cooper_sebat_iakoucheva_mooney_radivojac_2019,
    title={Pathogenicity and functional impact of non-frameshifting insertion/deletion variation in the human genome},
    volume={15},
    DOI={10.1371/journal.pcbi.1007112},
    number={6},
    journal={PLOS Computational Biology},
    author={Pagel, Kymberleigh A. and Antaki, Danny and Lian, AoJie and Mort, Matthew and Cooper, David N. and Sebat, Jonathan and Iakoucheva, Lilia M. and Mooney, Sean D. and Radivojac, Predrag},
    year={2019},
    pages={e1007112}
```
