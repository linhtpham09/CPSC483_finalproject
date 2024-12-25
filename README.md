# CPSC483_finalproject


Recommendation systems today are highly applicable to a wide range of industries, acting as a key component for companies to improve their businesses by elevating the customer experience. To make targeted recommendations, these algorithms rely on understanding the multidimensional relationships across users and products. By identifying patterns in user preferences, businesses can leverage these models to successfully appeal to the psychology of their customer base. 

After conducting a literature search on GNN algorithms applied to recommendation systems, we identified that there is room for exploration regarding the attention mechanism used by knowledge graph attention networks (KGAT) [1]. Thus, the overall goal of our project was to integrate the multi-hop attention mechanism [2] into a KGAT and analyze the performance of the algorithm in producing recommendations for customers on the Taobao dataset from PyTorch Geometric. To do so, we had to modify the existing Taobao dataset to address memory issues during the training steps. In the end, we ended up filtering for a much smaller and sparser Collaborative Knowledge Graph. Due to the severe memory issues, we were unable to evaluate the model performance on varying degrees of sparsity. Instead, we tested the effects of integrating multi-hop attention into the KGAT by comparing the model outputs when the architecture leveraged one hop, two hop, and three hop attention. 

After performing our tests, we saw that multi-hop attention did not improve model performance on the Taobao dataset. Furthermore, in general, the model provided better recommendations on the Amazon dataset. We propose that the reasons are most likely due to issues with the dataset (such as having few item features) and offer future steps to explore how modifying the attention mechanism in the KGAT will affect model performance. 

[1] Wang, Guangtao, et al. “Multi-Hop Attention Graph Neural Network.” arXiv.Org, 29 Sept. 2020, https://arxiv.org/abs/2009.14332v5.

[2] Ying, Rex, et al. “Graph Convolutional Neural Networks for Web-Scale Recommender Systems.” Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 974–83. arXiv.org, https://doi.org/10.1145/3219819.3219890.

## Navigating documents 

- exploration.ipynb : has the code for initial exploration and visualizations included in our project proposal 
- main.py : main file for replication (test and training)
- data : folder with kg_final, test, and train 
- data_processing : a folder holding our files for data preprocessing
- utility : a folder holding helper functions and architectures 


## Steps for replicating 

In order to replicate results using the default parameters (epoch = 10, batch size = 16, etc...) simply run ``python3 main.py`` in your terminal. 

However, in order to change the defaults, use --argument <value>. For example: 

```bash
python3 main.py --epoch 50 --alpha 0.1 --hop "three"
```
would run the  ``main.py`` file with 50 epochs, an alpha value of 0.1 and three hop attention. To see more default arguments, go to `utility/parser.py`


