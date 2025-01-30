
# %%
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import RandomNodeSplit
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from itertools import combinations
import time
from statistics import mean

# %%
# Load and preprocess the Iris dataset
def load_data(file_path):
    """
    Load the Iris dataset and preprocess the features and labels.

    Args:
        file_path (str): Path to the Iris dataset CSV file.

    Returns:
        x (torch.Tensor): Node features as a tensor.
        y (torch.Tensor): Node labels as a tensor.
    """
    iris_data = pd.read_csv(file_path)
    print("Unique species:", iris_data['Species'].unique())
    
    # Extract features and convert to tensor
    node_features = iris_data.iloc[:, 1:-1].values.astype(float)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Map species labels to integers and convert to tensor
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = torch.tensor([label_map[species] for species in iris_data['Species']], dtype=torch.long)
    
    return x, y

# %%
# Create a homogeneous graph
def create_graph(x, y):
    """
    Construct a homogeneous graph from node features and labels.

    Args:
        x (torch.Tensor): Node features.
        y (torch.Tensor): Node labels.

    Returns:
        Data: PyTorch Geometric Data object containing the graph.
    """
    edge_index_list = []
    for label in torch.unique(y):
        # get the indices of nodes that have this label
        nodes_in_class = (y == label).nonzero(as_tuple=True)[0]
        print(combinations(nodes_in_class, 2))
        # create combinations of nodes within the same class
        for (i, j) in combinations(nodes_in_class, 2):
            edge_index_list.append([i.item(), j.item()])
            edge_index_list.append([j.item(), i.item()])  # to add both directions for undirected graph
    # print(edge_index_list)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    print(edge_index)
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    print(graph_data)
    return graph_data

# %%
# Visualize the graph
def visualize_graph(graph_data):
    """
    Visualize the graph using NetworkX.

    Args:
        graph_data (Data): PyTorch Geometric Data object.
    """
    # convert the PyTorch Geometric graph to a NetworkX graph
    G = to_networkx(graph_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=graph_data.y.numpy(), cmap=plt.get_cmap('coolwarm'),
            node_size=500, edge_color='gray', font_size=10)
    plt.show()

# %%
# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialize the GCN model with two convolutional layers.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden layer units.
            out_channels (int): Number of output classes.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Log-softmax of the output predictions.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# %%
# Train and evaluate the GCN model
def train_and_evaluate(graph_data, batch_sizes, model, optimizer, epochs=10):
    """
    Train and evaluate the GCN model with different batch sizes.

    Args:
        graph_data (Data): PyTorch Geometric Data object.
        batch_sizes (list): List of batch sizes to evaluate.
        model (torch.nn.Module): GCN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs.

    Returns:
        dict: Results containing metrics for each batch size.
    """
    results = {}
    start_time = time.time()

    for batch_size in batch_sizes:
        # Initialize loaders
        train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                      num_neighbors=[0], batch_size=batch_size,
                                      directed=False, shuffle=False)
        test_loader = NeighborLoader(graph_data, input_nodes=graph_data.test_mask,
                                     num_neighbors=[0], batch_size=batch_size,
                                     directed=False, shuffle=False)

        epoch_losses, test_accuracies, train_accuracies = [], [], []

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, train_correct, train_total = 0, 0, 0

            for batch in train_loader:
                training_batch_x = batch.x[batch['train_mask'],:]
                # print('Current Train Batch Size', training_batch_x.shape)
                # identify training nodes for ex: total no. of train nodes in the current is 22
                train_nodes = batch.train_mask.nonzero(as_tuple=True)[0]
                # now here we want to consider inly those edges that are connected to train_nodes
                # use torch.isin to check if elements of edge_index[0] and edge_index[1] are in train_nodes
                train_edge_mask = torch.isin(batch.edge_index[0], train_nodes) & torch.isin(batch.edge_index[1], train_nodes)
                # suppose there are 488 connections (edge_index = [2, 488]) so train_edge_mask.sum().item() = 164 accounts for connection 
                # where source and target nodes both are in training set. Likewise, val_edge_mask.sum().item() = 32
                # there are 32 edges connecting nodes in the validation set, and 10 edges within the test set.
                # remaining edges in edge_index account for connections that span across different sets 
                # training-to-validation, training-to-test which explains 488 connections. 
                # filter edge_index based on this mask
                # print('Total edges/connections that includes only train nodes', train_edge_mask.sum().item())
                training_edge_index = batch.edge_index[:, train_edge_mask]

                # the below code is added to resolve out-of-bound error:
                # create a mapping from old indices to new indices
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(train_nodes)}
                # apply the mapping to `training_edge_index`
                reindexed_edge_index = torch.tensor([
                    # training_edge_index[0]: target node
                    [node_mapping[node.item()] for node in training_edge_index[0]],
                    # training_edge_index[1]: source node
                    [node_mapping[node.item()] for node in training_edge_index[1]]
                ], dtype=torch.int64)

                # print('Edge index: connections where source and target are in training sets ', training_edge_index)
                # print('reindexed_edge_index', reindexed_edge_index)

                optimizer.zero_grad()
                out = model(training_batch_x, reindexed_edge_index)
                # print('Output Shape:', out.shape)
                # extract only the labels corresponding to training nodes
                train_y = batch.y[batch.train_mask]
                loss = F.cross_entropy(out, train_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate training accuracy
                pred = out.argmax(dim=1)
                train_correct += (pred == train_y).sum().item()
                train_total += train_y.size(0)

            epoch_losses.append(total_loss / len(train_loader))
            train_accuracies.append(train_correct / train_total)

            # Evaluate on test set
            # testing at each epoch
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    test_batch_x = batch.x[batch['test_mask'],:]
                    # print('Current Test Batch Size', test_batch_x.shape)
                    test_nodes = batch.test_mask.nonzero(as_tuple=True)[0]
                    test_edge_mask = torch.isin(batch.edge_index[0], test_nodes) & torch.isin(batch.edge_index[1], test_nodes)
                    test_edge_index = batch.edge_index[:, test_edge_mask]
                    test_node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(test_nodes)}
                    test_reindexed_edge_index = torch.tensor([
                        # target node
                        [test_node_mapping[node.item()] for node in test_edge_index[0]],
                        # source node
                        [test_node_mapping[node.item()] for node in test_edge_index[1]]
                    ], dtype=torch.int64)

                    # print('Edge index: connections where source and target are in test sets ', test_edge_index)
                    # print('reindexed_edge_index', reindexed_edge_index)
                    optimizer.zero_grad()
                    out = model(test_batch_x, test_reindexed_edge_index)
                    # print('Test Output Shape:', out.shape)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y[batch.test_mask]).sum().item()
                    total += batch.test_mask.sum().item()
            test_accuracy = correct / total if total > 0 else 0
            test_accuracies.append(test_accuracy)

        end_time = time.time()
        results[batch_size] = {
            'avg_train_loss': mean(epoch_losses),
            'final_train_accuracy': train_accuracies[-1],
            'final_test_accuracy': test_accuracies[-1],
            'training_time': end_time - start_time
        }

    return results

# %%
# Main script
file_path = 'data/Iris.csv'
x, y = load_data(file_path)
graph_data = create_graph(x, y)
visualize_graph(graph_data)

# Split the graph into training, validation, and test sets
np.random.seed(3332)
splitter = RandomNodeSplit(split='train_rest', num_val=20, num_test=20)
graph_data = splitter(graph_data)

# Initialize the model and optimizer
model = GCN(in_channels=4, hidden_channels=16, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train and evaluate the model
batch_sizes = [8, 16, 30]
results = train_and_evaluate(graph_data, batch_sizes, model, optimizer, epochs=10)

# Display results
for batch_size, metrics in results.items():
    print(f"\nBatch Size: {batch_size}")
    print(f"Average Training Loss: {metrics['avg_train_loss']:.4f}")
    print(f"Final Training Accuracy: {metrics['final_train_accuracy']:.4f}")
    print(f"Final Test Accuracy: {metrics['final_test_accuracy']:.4f}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
# %%
