# CPSC483_finalproject

## Navigating documents 

- 1_LP_exploration.ipynb : has the code for initial exploration and visualizations 
- 2_kgat : this folder has all of the documents I used to replicate the KGAT paper and understand it
- 3_magna: this folder has all of the documents I used to replicate the MAGNA paper and understand it 
- official_code: this folder holds some of the most important source code from the original papers
- scratch: this folder has some extra .py files and notebooks when I needed to experiment 


## Steps for replicating KGAT Paper 

 1. Dataset Preparation
 Need to preprocess datasets to construct a **user-item interaction graph** and **knowledge graph** 
 - **user-item interaction graph** : bipartite graph where users and items are connected based on interactions (e.g. clicks, purchases)
 - **knowledge graph**: Entities and relations between items(eg. an item could be a movie, which has directors, genres, etc)

The KGAT paper preprocesses their datasets using [KB4Rec](https://direct.mit.edu/dint/article/1/2/121/27497/KB4Rec-A-Data-Set-for-Linking-Knowledge-Bases-with). They do this process in order to populate their dataset with more knowledge of the products. This allows them to create more relationships between users and items. We both cannot do and do not need to do this process. This is because the datasets the paper uses links to real world names (i.e. "Titanic" is a real movie and more information can be found on it). However, the Taobao dataset only has id numbers, so we could not do this process in the same way. 

The Taobao dataset does give us a comprehensive look into the users, items, and respective entities. 


$\square$ copy code for Amazon-Book preprocessing 

Notes:
- train.txt and test.txt represent user-item interaction data 
- kg_final.txt is knowledge graph data (head, relation, tail)
- item

$\square$ do same for Taobao preprocessing 

2. KGAT Model 
Exploits high order relations in an end-to-end fashion. 

    a. Embedding layer: parameterizes each node as a vector by preserving the structure of the Collaborative Knowledge Graph 

    b. Attentive embedding propagation layers: recursively propagates embeddings from a node's neighbors to update its representation and employ knowledge aware attention [!where we change!] to learn the weight of each neighbor during a propagation 

    c. prediction layer: which aggregates the representation of a user and item from all propagation layers, and outputs the predicted matching score 


3. Loss function - Bayesian Personalized Ranking (BPR) loss and knowledge graph regularization loss 
4. Evaluation metrics - Recall@K and NDCG@K to evaluate the rec performance on data 


