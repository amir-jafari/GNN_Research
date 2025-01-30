'''
Graph-Based Classification of Iris Species Using Graph Convolutional Networks (GCNs)

This script performs node classification on the Iris dataset, a popular dataset in machine learning. 
Here, we treat each data point (flower sample) as a node and build a graph based on the similarities 
among the nodes' features (i.e., petal and sepal measurements). By constructing a k-nearest neighbors 
(k-NN) graph, we create edges between nodes to represent these relationships. 

We use PyTorch Geometric to define and implement a two-layer Graph Convolutional Network (GCN) that learns
to classify each node (flower sample) into one of three Iris species: Setosa, Versicolor, or Virginica. 
The code also includes graph creation, data loading, and a training loop with evaluation. 

This script demonstrates key concepts in graph neural networks:
1. Data transformation and graph construction.
2. Training and evaluating a GCN for node classification.
3. The use of various training parameters, including batch size, and an exploration of their impact on performance.

Identifying the Graph Type: 
- Homogeneous graph: Same type of nodes and edges (e.g., all nodes are flowers, and all edges represent similarity).
- Heterogeneous graph: Different types of nodes and/or edges (e.g., flowers, species, and different relationships)
- The Iris dataset can be represented as a homogeneous graph where each node is a data point (an iris flower sample), 
and edges represent "similar", with all nodes and edges being of a single type.
'''

# %%
import torch
from torch_geometric.data import Data
# from sklearn.neighbors import kneighbors_graph 
from torch_geometric.nn import knn_graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
# from sklearn.metrics import pairwise_distances
from torch_geometric.transforms import RandomNodeSplit
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from itertools import combinations

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# %%
file_path = 'data/Iris.csv'
iris_data = pd.read_csv(file_path)
print(iris_data)
print(iris_data['Species'].unique())


# %%
# determining k using knn from scikit-learn
# X = iris_data.drop(columns=['Id', 'Species'])
# label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# y = iris_data['Species'].map(label_map)
# print(X)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# n = 10
# k_values = range(1, n + 1)
# error_rates = []

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     error_rate = 1 - accuracy_score(y_test, y_pred)
#     error_rates.append(error_rate)

# # Plot Error Rate vs. k
# plt.figure(figsize=(8, 6))
# plt.plot(k_values, error_rates, marker='o', linestyle='-', color='b')
# plt.title('Error Rate vs. k (Elbow Method)')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Error Rate')
# plt.xticks(k_values)
# plt.grid()
# plt.show()



# %%
# extract node features (all rows, without the 'Id' and 'Species' columns)
node_features = iris_data.iloc[:, 1:-1].values.astype(float)
print(node_features)
print(node_features.shape)
# convert the features into a tensor
x = torch.tensor(node_features, dtype=torch.float)
print(x)
print(x[0])
print(x[0][0])
print(x[0][2])
print(x.shape)

# %%
# def calculate_ssd(k, x):
#     distances = pairwise_distances(x.numpy(), x.numpy())
#     ssd = 0
#     for i in range(len(x)):
#         nearest_distances = np.sort(distances[i])[1:k+1]
#         print(nearest_distances)
#         ssd += np.sum(nearest_distances**2)
#     return ssd

# k_values = range(2, 20)
# ssd_values = [calculate_ssd(k, x) for k in k_values]
# print(ssd_values)

# plt.figure(figsize=(8, 6))
# plt.plot(k_values, ssd_values, marker='o', linestyle='-')
# plt.xlabel('k (Number of Nearest Neighbors)')
# plt.ylabel('Sum of Squared Distances (SSD)')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# %%
# k = 6
# edge_index = knn_graph(x, k=k, loop=False)
# print(edge_index)
# print(edge_index.shape)

# %%
# mapping labels to numeric
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# convert labels into tensor
y = torch.tensor([label_map[species] for species in iris_data['Species']], dtype=torch.long)
print(y)
print(y[0])
print(y.shape)

# %%
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

# %%
# convert the PyTorch Geometric graph to a NetworkX graph
G = to_networkx(graph_data, to_undirected=True)
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color=graph_data.y.numpy(), cmap=plt.get_cmap('coolwarm'),
        node_size=500, edge_color='gray', font_size=10)
plt.show()

# %%
splitter = RandomNodeSplit(split='train_rest', num_val = 20, num_test=20)
graph_data = splitter(graph_data)
print(graph_data)
print(graph_data.train_mask)
print(graph_data.val_mask)
print(graph_data.test_mask)
print(graph_data.train_mask.sum().item()) 
print(graph_data.val_mask.sum().item())
print(graph_data.test_mask.sum().item())


# %%
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # first GCN layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # second GCN layer (output layer)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # since here we are using two GCN layers, preprocessing steps (message aggregation) will be done twice for each node.
        # GCNConv inherits MessagePassing class 

    def forward(self, train_x, train_edge_index):
        x, edge_index = train_x, train_edge_index
        # node features are transformed at each GCN layer
        # first layer: apply convolution and ReLU activation
        # print('Size of xxxxxxxxxxxxx', x.shape)
        x = self.conv1(x, edge_index)
        # print('Output from conv1', x.shape)
        x = F.relu(x)
        # print('Output after applying  ReLU', x.shape)
        # second layer: apply convolution and softmax activation for classification (since we have more than two class)
        x = self.conv2(x, edge_index) 
        # print('Output from conv2', x.shape)
        return F.log_softmax(x, dim=1)
    
model = GCN(in_channels=4, hidden_channels=16, out_channels=3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# %%
# Only for an epoch with batch size 5 (Just to see how it works)
import numpy as np
np.random.seed(3332)
train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                  num_neighbors=[3], batch_size=7,
                                  directed=False, shuffle=False)

print('Length of train_loader', len(train_loader))

print(train_loader)
for epoch in range(1, 2):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    counter = 0
    for batch in train_loader:
        counter = counter + 1
        print(counter)
        # print('Current batch', batch)
        # print('Len of current batch', len(batch))
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
        # print('Train Y - Labels corresponding to training nodes:', train_y)
        loss = F.cross_entropy(out, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate training accuracy
        pred = out.argmax(dim=1)
        train_correct += (pred == train_y).sum().item()
        train_total += train_y.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = train_correct / train_total if train_total > 0 else 0
    print('Train Accuracy', train_accuracy)


# %%
import time
from statistics import mean

start_time = time.time()
epoch_losses = []
test_accuracies = []
train_accuracies = []

batch_sizes = [8, 16, 30]
results = {}
for batch_size in batch_sizes:
    # initialize loaders
    train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                  num_neighbors=[6], batch_size=batch_size,
                                  directed=False, shuffle=True)

    val_loader = NeighborLoader(graph_data, input_nodes=graph_data.val_mask,
                                num_neighbors=[6], batch_size=batch_size,
                                directed=False, shuffle=False)

    test_loader = NeighborLoader(graph_data, input_nodes=graph_data.test_mask,
                                 num_neighbors=[6], batch_size=batch_size,
                                 directed=False, shuffle=False)
    
    print(train_loader)
    for epoch in range(1, 11):
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
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
                # print('Train Y - Labels corresponding to training nodes:', train_y)
                loss = F.cross_entropy(out, train_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate training accuracy
                pred = out.argmax(dim=1)
                train_correct += (pred == train_y).sum().item()
                train_total += train_y.size(0)

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_accuracies.append(train_accuracy)
            
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
    training_time = end_time - start_time

    results[batch_size] = {
        'avg_train_loss': mean(epoch_losses),
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1],
        'training_time': training_time
    }

for batch_size, metrics in results.items():
    print(f"\nBatch Size: {batch_size}")
    print(f"Average Training Loss: {metrics['avg_train_loss']:.2f}")
    print(f"Final Training Accuracy: {metrics['final_train_accuracy']:.2f}")
    print(f"Final Test Accuracy: {metrics['final_test_accuracy']:.2f}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")

