'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse


def parse_args():
    """
python Main.py --model_type kgat --alg_type bi --dataset amazon-book 
--regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64
 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 
 --pretrain -1 --batch_size 1024 --node_dropout [0.1] 
 --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
"""

    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument('--model_type', nargs='?', default='kgat',
                        help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--alg_type', nargs='?', default='bi',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--dataset', nargs='?', default='taobao_scarce',
                        help='Choose a dataset from {amazon-book,taobao_scarce, taobao_full}')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--layer_size', nargs='?', default='[64,32,16]',
                        help='Output sizes of every layer')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='CF Embedding size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=10, #paper uses 1000 epochs, but we don't have enough time
                        help='Number of epoch.')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Interval of evaluation.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='CF batch size.')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--use_att', type=bool, default=True,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=True,
                        help='whether using knowledge graph embedding')
    parser.add_argument('--loss_type', nargs='?', default='bpr',
                        help='Specify a loss type from {bpr, ce}.')
    parser.add_argument('--hop', type = str, default='one',
                        help='Specify a loss type from {one,two,three}.')
    parser.add_argument('--alpha', type = float, default=0.25,
                        help='Please type a number for the alpha in attention diffusion')
    
#other defaults not specified 

    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--kge_size', type=int, default=32,
                        help='KG Embedding size.')
    parser.add_argument('--batch_size_kg', type=int, default=16,
                        help='KG batch size.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--l1_flag', type=bool, default=True,
                        help='Flase: using the L2 norm, True: using the L1 norm.')

    return parser.parse_args()
