# MAGNA Notes 

These notes are primarily for the attention mechanism in MAGNA

1. Compute attention score of an edge $(v_i, r_k, v_j)$ is computed by the following: 
$$ s_{i,k,j}^l = \text{LeakyReLU} (v_a^l \text{tanh}(W_h^lh_i^l || W_t^lh_j^l||W_r^lr_k^l))$$ 

2. Softmax $a_{ij} = \text{softmax}(\pi(h_A, h_B))$

3. Multi Hop Attention Diffusion 

$$A' = \sum^K \theta^k A^k$$
where k is number of hops and $a^k$ is decay function 

4. Message passing 

$$h_A^{new} =\sum_{j \in \text{neighbors}}A'_{ij}h_j$$

But we can instead use the following theory 

$$Z ^{(0)} = H^l$$ 
$$ Z^{k+1} = (1-a) AZ^k + aZ^{(0)}$$

for message propagation where Z refers to the next layer. 