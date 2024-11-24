import os
import torch
print("Using torch", torch.__version__)
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch_geometric.data import Data
from torch_geometric import nn
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split 
from torch_geometric.datasets import AmazonBook
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import HeteroConv, GCNConv

dataset = AmazonBook(root = './amazonbook')
data = dataset[0]
print('read data')

print("Node types:", data.node_types)

e1 = data.edge_types[0]
e2 = data.edge_types[1]
print("Edge types:", e1, e2)
print('user nodes: ', data['user'].num_nodes)
print('book nodes: ', data['book'].num_nodes)

# Function to filter edges based on node subsets
def filter_edges(edge_index, valid_src, valid_dst):
    src_mask = torch.isin(edge_index[0], valid_src)
    dst_mask = torch.isin(edge_index[1], valid_dst)
    edge_mask = src_mask & dst_mask
    return edge_index[:, edge_mask]
print("##############################################################################")
print("Downloaded Data")
print("##############################################################################")
##################################################################################
selected_books = torch.randint(0, data['book'].num_nodes, (5000,))
selected_users = torch.randint(0, data['user'].num_nodes, (5000,))


# Step 2: Filter `user -> book` edges
user_book_edges = data[e1].edge_index
filtered_user_book_edges = filter_edges(user_book_edges, selected_users, selected_books)

# Step 3: Identify the remaining valid users and books
valid_users = torch.unique(filtered_user_book_edges[0])
valid_books = torch.unique(filtered_user_book_edges[1])

# Step 4: Filter `book -> user` edges
book_user_edges = data[e2].edge_index
filtered_book_user_edges = filter_edges(book_user_edges, valid_books, valid_users)

# Step 5: Finalize the subgraph with sizes 
final_users = torch.unique(filtered_book_user_edges[1])
final_books = torch.unique(filtered_book_user_edges[0])

# Step 6: Re-filter edges to match final users and books
filtered_user_book_edges = filter_edges(user_book_edges, final_users, final_books)
filtered_book_user_edges = filter_edges(book_user_edges, final_books, final_users)

# Step 7: Create the subgraph
subset_data = HeteroData()
subset_data['user'].num_nodes = final_users.size(0)
subset_data['book'].num_nodes = final_books.size(0)
subset_data['user', 'rates', 'book'].edge_index = filtered_user_book_edges
subset_data['book', 'rated_by', 'user'].edge_index = filtered_book_user_edges

# Verify the subgraph
print(f"Number of users: {subset_data['user'].num_nodes}")
print(f"Number of books: {subset_data['book'].num_nodes}")
print(f"User -> Book edges: {subset_data['user', 'rates', 'book'].edge_index.shape[1]}")
print(f"Book -> User edges: {subset_data['book', 'rated_by', 'user'].edge_index.shape[1]}")

feature_dim = 16  # Choose a smaller dimension to save memory
subset_data['user'].x=torch.eye(subset_data['user'].num_nodes)
subset_data['book'].x = torch.eye(subset_data['book'].num_nodes)
print("##############################################################################")
print("Finished subsetting")
print("##############################################################################")
##########################################Splitting into train, test, and validate data 
train_data = HeteroData()
val_data = HeteroData()
test_data = HeteroData()

# Ensure num_nodes is set in subset_data
for node_type in subset_data.node_types:
    if 'num_nodes' not in subset_data[node_type] or subset_data[node_type].num_nodes is None:
        subset_data[node_type].num_nodes = int(subset_data[node_type].x.shape[0])

# Set x_dict globally for all subsets
train_data.x_dict = subset_data.x_dict
val_data.x_dict = subset_data.x_dict
test_data.x_dict = subset_data.x_dict

for edge_type in subset_data.edge_types:
    # Edge splitting
    edge_index = subset_data[edge_type].edge_index.T
    train_edges, test_edges = train_test_split(edge_index, test_size=0.2, random_state=42)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.25, random_state=42)

    # Convert back to tensors
    train_edges = torch.tensor(train_edges).T
    val_edges = torch.tensor(val_edges).T
    test_edges = torch.tensor(test_edges).T

    # Assign edges to corresponding HeteroData objects
    train_data[edge_type].edge_index = train_edges
    val_data[edge_type].edge_index = val_edges
    test_data[edge_type].edge_index = test_edges

    # Assign edge_label_index
    train_data[edge_type].edge_label_index = train_edges
    val_data[edge_type].edge_label_index = val_edges
    test_data[edge_type].edge_label_index = test_edges

    # Assign edge_label (positive class for all edges)
    train_data[edge_type].edge_label = torch.ones(train_edges.shape[1])
    val_data[edge_type].edge_label = torch.ones(val_edges.shape[1])
    test_data[edge_type].edge_label = torch.ones(test_edges.shape[1])

    # Negative sampling
    num_nodes = max(subset_data[edge_type[0]].num_nodes, subset_data[edge_type[-1]].num_nodes)
    neg_samples = negative_sampling(
        edge_index=train_edges,
        num_nodes=num_nodes,
        num_neg_samples=min(train_edges.shape[1], 1000)
    )
    train_data[edge_type].neg_edge_index = neg_samples
    train_data[edge_type].neg_edge_label = torch.zeros(neg_samples.shape[1])

    print(f"Edge type {edge_type}:")
    print("Number of nodes:", subset_data[edge_type[0]].num_nodes)
    print("Train edges:", train_edges.shape[1])
    print("Validation edges:", val_edges.shape[1])
    print("Test edges:", test_edges.shape[1])
    print("Edge labels in train:", train_edges.shape[1])


print("##############################################################################")


feature_dim = 16  # Set feature dimension as needed
for node_type in train_data.node_types:
    if 'x' not in train_data[node_type]:
        num_nodes = train_data[node_type].num_nodes
        train_data[node_type].x = torch.randn(num_nodes, feature_dim)
        val_data[node_type].x = train_data[node_type].x
        test_data[node_type].x = train_data[node_type].x

for node_type in train_data.node_types:
    print(f"Node type: {node_type}, x exists: {'x' in train_data[node_type]}")    
for edge_type in train_data.edge_types:
    print(f"Edge type: {edge_type}, edge_index exists: {'edge_index' in train_data[edge_type]}")

print("##############################################################################")
print("Finished splitting data into train, validation, and test ")
print("##############################################################################")
##########################################-PIPELINE-#########################################


from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGCN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.convs = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')  # Aggregate messages across edge types

        self.out_conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Perform message passing for each edge type
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}

        # Perform the final layer of message passing
        x_dict = self.out_conv(x_dict, edge_index_dict)
        return x_dict

##################################################################################

model = HeteroGCN(subset_data.metadata(), hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

##################################################################################

def compute_similarity(node_embs, edge_index):
    result = 0
    result = (node_embs[edge_index[0]] *node_embs[edge_index[1]]).sum(dim=1)
    return result

def train(model, data, optimizer, loss_fn):

    loss = 0

    # ARGS: 
    #model - the heterogeneous GCN model: HeteroGCN 
    #data - HeteroData object with x_dict, edge_index_dict and target labels zz
    #optimizer: Adam 
    #criterion: Loss function, Binary Cross Entropy Loss
    model.train() 
    optimizer.zero_grad() 

    #forward pass: pass node features and edge indices through the model 
    out_dict = model(data.x_dict, data.edge_index_dict)

    #compute loss for all edge types (eg. link prediction)
    total_loss = 0 
    for edge_type in data.edge_types: 
        if 'edge_label' in data[edge_type]: #check if edge labels exist for this type
            pred = out_dict[edge_type[-1]] # get predictions for target node type
            target = data[edge_type].edge_label#true labels for this edge type
            total_loss += loss_fn(pred, target)


    total_loss.backward() 
    optimizer.step()

    return total_loss.item()


#########################################    #########################################


@torch.no_grad()
def test(model, data):
    model.eval()

    total_auc = 0 
    total_edges = 0 

    for edge_type in data.edge_types: 
        if 'edge_label' in data[edge_type]:#ensure labels exist 
            #perform message passing and get predictions for this edge type 
            out = model(data.x_dict, data.edge_index_dict)
            node_embeddings = out[edge_type[-1]]# target node type embeddings

            #compute similarity for edges in edge_label_index
            edge_label_index = data[edge_type].edge_label_index
            predictions = compute_similarity(node_embeddings, edge_label_index).view(-1).sigmoid()

            #compute AUC for this edge type 
            edge_label = data[edge_type].edge_label
            auc = roc_auc_score(edge_label.cpu().numpy(), predictions.cpu().numpy())
            
            # Weighted aggregation of AUC scores
            num_edges = edge_label.size(0)
            total_auc += auc * num_edges
            total_edges += num_edges

    # Compute the final AUC (weighted average across edge types)
    return total_auc / total_edges if total_edges > 0 else 0

print("##############################################################################")
print("Starting training")

epochs = 50

best_val_auc = final_test_auc = 0
for epoch in range(1, epochs + 1):
    loss = train(model, train_data, optimizer, loss_fn)
    valid_auc = test(model, val_data)
    test_auc = test(model, test_data)
    if valid_auc > best_val_auc:
        best_val_auc = valid_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {valid_auc:.4f}, Test: {test_auc:.4f}')

print("Finished!")