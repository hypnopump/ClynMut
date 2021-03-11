## Predicting the Clynical Relevance of Genome Mutations

To be a next-generation DL-based phenotype prediction from genome mutations. Will use sota NLP and structural techniques. 

### Planned approach:

Planned input features will likely be: 
* 3D structure
* NLP embeddings


Planned modules will likely be: 
* Standard linear layers
* 3D learning module (GVP, E(n)-equivariants, ...)
* hierarchical classification (probably gaussian processes, but not sure yet. could do mlps for now as well)

The main idea is for the model to learn the prediction in an end-to-end fashion. A sample architecture:

```
Sequence ------------+--> 3D_structure --> 3D_module --+                                      +--> ?
|                    |                                 |                                      +--> ?
|                    |                                 +--> Joint_module --> Hierarchical CLF +--> ?
|                    |                                 |                                      +--> ?
+--> NLP_embeddings -+-------------> Embedding_module -+                                      +--> ?
```


#### References:

Projects with similar aims (year : name : paper_link : poster_link) ranked by paper results (interpreted by us): 

* 2017 : MutPred2 Preprint: https://doi.org/10.1038/s41467-020-19669-x : http://mutpred.mutdb.org/
* 2019 : MutPred Paper : https://pdfs.semanticscholar.org/b1c4/31717cf470634bfb5faca0c0ec9d3bd5ec66.pdf
* 2020 : NLP-SNPPred : https://ieeexplore.ieee.org/document/9175781/ : 


### Datasets: 

* referenced in NLP-SNPPred : https://ieeexplore.ieee.org/document/9175781/
    * training:
        * OncoKB : ()
    * validation: 
        * VariBench (three positive/pathogenic datasets: TP53, ClinVar and DoCM)
        * CIViC (expert-crowd-sourced knowledge base of variants in cancer. We only considered those examples that were labeled as "Likely Pathogenic”, “Poor Outcome” or “Negative”)

* referenced in MutPred : https://pdfs.semanticscholar.org/b1c4/31717cf470634bfb5faca0c0ec9d3bd5ec66.pdf
    * training:
        * Human Gene Mutation Database (HGMD)
        * Genome Aggregation Database (gnomAD)
    * validation:
        * COSMIC : (Catalogue Of Somatic Mutations In Cancer)
        * dbCID (DataBase of Cancer Driver InDel) : 
        * affected by autism pectrum disorder (ASD) from the REACH Project and the Simons Simplex Collection


### Contribute
Hey there! New ideas are welcome: open/close issues, fork the repo and share your code with a Pull Request.
Clone this project to your computer:
 
`git clone https://github.com/EricAlcaide/ClynMut`
 
By participating in this project, you agree to abide by the thoughtbot [code of conduct](https://thoughtbot.com/open-source-code-of-conduct)
 
### Meta
 
* **Author's GitHub Profile**: [Eric Alcaide](https://github.com/hypnopump/)
* **Twitter**: [@eric_alcaide](https://twitter.com/eric_alcaide)
* **Email**: ericalcaide1@gmail.com
