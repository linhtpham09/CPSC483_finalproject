import pandas as pd 
import numpy as np 
from argparse import Namespace 


import importlib
from utility import load_data

# Force reload
importlib.reload(load_data)

from utility.load_data import Data

#python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True


args =Namespace(batch_size=512, 
                adj_type = 'bi', 
                mess_dropout = [0.1,0.1,0.1],
                node_dropout = [0.1],
                layer_size = [64,32,16]
                )

data = Data(args, path = 'data')
"""
Data is a class object that helps process the data. In this example, 
I have my folder set up so that all my data is in txt files in a data folder. 
First, it takes the the path file to load in train.txt, test.txt, and kg_final.txt 

Note: for Taobao, we will need to make our own kg_final.txt
"""
#data._print_data_info() same thing as print(data)
"""
output: 
[n_users, n_items]=[70679, 24915]
[n_train, n_test]=[652514, 193920]
[n_entities, n_relations, n_triples]=[113487, 39, 2557746]
[batch_size, batch_size_kg]=[512, 2007]
<utility.load_data.Data object at 0x104895b90>
"""

##note to Kerui: encourage you to try out different print functions to learn 
##what class objects do! here are some below 
# print("Train Data:", data.train_data)
# print("Test Data:", data.test_data)
# print("Knowledge Graph:", data.kg_data)

train = data.train_data 
test= data.test_data
print(train[:5]) 
print(test[:5])

users, pos_items, neg_items = data._generate_train_cf_batch() #from load_data.py
print("Users:", users[:5])
print("Positive Items:", pos_items[:5])
print("Negative Items:", neg_items[:5])
"""
[write explanation of _generate_train_cf_batch()]
This function helps us create user-item interaction batches for training


Collectively, train, test, users, pos_items, and neg_items represent the User-Item Bipartite Graph. Positive edges are actual user-item interactions from train and test. Negative edges are non-interactions. 
The paper specifically writes G = {(u,y_{ui}, i) | u \in U, i \in I}

Each train and test gives the user and item they interacted with, a positive edge. We can think of this as follows 
(users == train.users, where pos_items| neg_items == {some interaction},  train.items ) to get the same triplet represented above 
and vice versa for test 
Users          Items
   1   ────>   101   (positive)
   2   ────>   102   (positive)
   1   ────>   201   (negative)
   2   ────>   202   (negative)

"""

#Phase II 
from utility.loader_kgat import KGAT_loader
kgat_loader = KGAT_loader(args = args, path = 'data')
heads, pos_r_batch, pos_t_batch, neg_t_batch= kgat_loader._generate_train_A_batch() #from loader_kgat 

"""
[explain what _generate_train_A_batch does]
"""

print("heads: ",heads[:5])
print("pos_r_batch: ",pos_r_batch[:5])
print("pos_t_batch: ",pos_t_batch[:5])
print("neg_t_batch: ",neg_t_batch[:5])
print('kg done')