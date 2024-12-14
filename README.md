# CPSC483_finalproject

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


