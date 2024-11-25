

# Understanding 

KGAT Paper 
## 1. Preprocessing Data into a collaborative knowledge graph (CKG)

``load_data.py``
- focus: Collaborative Filtering 
- Input files: train.txt, test.txt, user_list.txt, item_list.txt
- primary outputs: user-item interactions and training batches
- no graph structures created
- training batch types: user item batches (CF)
- Main responssibility: load user-item interactions in train.txt and test.txt 
- load knowledge graph kg_final.txt 
- generate data structures: 

    a. train_user_dict & test_user_dict: map users to their interacted items
    b. kg_dict: map heads to (tail, relation) tuples for the knowledge graph 

- Functions: 
    - _load_ratings: load interaction data into user-item mappings
    - _load_kg: load knowledge graph and preprocess them into dictionaries for efficient access 
    - _generate_train_cf_batch: gemerate user_item batches for collaborative filtering training


 ``loader_kgat.py`` 
extends load_data.py with graph specific preprocessing for KGAT 
- focus: Knowledge Graph 
- Input Files: kg_final.txt, relation_list.txt 
- Primary output: graph structures and KG embedding batches
- graph structures: adjacency and Laplacian matrices
- training batch tpes: Knowledge graph triples/embedddings (KGE)

- Sparse Graph Construction 

    - _get_relation_adj_list: Build adjacent matrices for both user-item and knolwedge graph relationships 
    - _get_relational_lap_list: Generate normalized Laplacian matrices (used in GNN for message passing)
- Triple Generation
    - generate_train_batch: Generate user-item pairs for collaborative filtering 
    - generate_train_A_batch: Generate knowledge graph triples with negative sampling
- Output Graph Data 
    - Adjacency and Laplacian matrices
    - Head, relation, tail lists for TransR-based graph embeddings 


## 2. Build KGAT model
 ``KGAT.py``

Purpose: Define KGAT architecture and training logic 

- Phase I (Collaborative filtering): Train CF embeddings using collaborative filtering loss
- Phase II (Knowledge graph embedding): Train KG embeddings using TransR and optimize with BPR loss
- Attentive Laplacian Update: refine adjacency matrix using KG attention scores 

Functions:
- _build_weights: Initialize embeddings for users, items, entities, and relations
- _build_model_phase_II: Implements TransR to map entitites and relations into relation-specific spaces
- _build_loss_phase_II: Optimize KG embeddings with BPR loss 

## 3. Evaluate Model 

``batch_test.py`` computes ranking metrics

Functions: 
- ranklist_by_heapq & ranklist_by_sorted: 
    - rank items for users based on predicted scores
    - evaluate ranking metrics (ex. precision, recall, NDCG, AUC)
- test_one_user: Evaluate predictions for a single user by comparing predicted rankings with ground truth 
- test: main testing function that evaluates the model for all users using metrics defined in ``metrics.py``



## 4. Train KGAT 
- execute ``Main.py`` with appropriate cl arguments for dataset and hyperparameters 

- Phase I: Collaborative filtering 
    - Generate user and item embeddings using graph neural networks
    - optimize CF specific embeddings with BPR loss 
- Phase II: Knowledge graph embeddings using TransR 
    - Generate entity and relation embeddings
    - use a modified BPR loss to optimize KG embeddings
- Attentive Laplacian Update: refind adjacency matrix using KG attention scores 

Functions: 
- _build_weights: Initialize embeddings for users, items, entities, and relations
- _build_model_phase_II: Implements TransR to map entities and relations into relation-specific spaces
- _build_loss_phase_II: Optimize KG embeddings with BPR loss 