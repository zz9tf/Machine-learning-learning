import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GraphConv, 
    GCNConv, 
    SGConv,
    AGNNConv,
    ResGatedGraphConv,
    GINEConv,
    global_mean_pool, 
    global_add_pool, 
    global_max_pool
)
from torch_geometric.data import Data, Dataset, DataLoader

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_node_labels=10, embedding_dim=3):
        super(GNN, self).__init__()
        
        # Node label embedding (maps integer labels to vectors)
        # self.node_embedding = torch.nn.Embedding(num_node_labels, embedding_dim)
        
        # Adjust first layer to take embedding dimension as input
        # self.conv1 = GCNConv(embedding_dim, hidden_channels)  # Input is now embedding dimension
        self.conv2 = GCNConv(1, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Linear layer after pooling
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # x shape is [num_nodes, 1] containing integer node labels
        
        # Convert node labels to long and remove the feature dimension
        # node_labels = x.long().squeeze(-1)
        
        # Create embeddings from node labels [num_nodes, embedding_dim]
        # x = self.node_embedding(node_labels)
        
        # Graph convolution with embeddings
        # x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Apply global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Final classifier
        x = self.lin(x)
        
        return x

# Custom dataset class
class GraphDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None):
        self.root = root
        self.mode = mode
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load adjacency matrix directly
        self.edges = np.loadtxt(os.path.join(root, 'A.txt'), delimiter=',', dtype=np.int64) - 1  # -1 because indices should be 0-based
        
        # Load graph indicators directly
        indicator_file = f'{mode}_graph_indicator.txt'
        indicators_path = os.path.join(root, indicator_file)
        
        # First count the columns in the file
        with open(indicators_path, 'r') as f:
            first_line = f.readline().strip()
            num_columns = len(first_line.split(','))
        
        self.indicators = np.loadtxt(indicators_path, delimiter=',', dtype=np.int64)
        self.node_indices = self.indicators[:, 0]
        self.graph_ids = self.indicators[:, 1]
        self.node_labels = self.indicators[:, 2]
        
        # Count the number of unique node labels for embedding
        self.num_node_labels = int(np.max(self.node_labels)) + 1
        print(f"Found {self.num_node_labels} unique node labels")
        
        # Get unique graph IDs
        self.unique_graph_ids = np.unique(self.graph_ids)
        
        if mode == 'train':
            # Load graph labels directly
            labels_file = f'{mode}_graph_labels.txt'
            labels_path = os.path.join(root, labels_file)
            graph_labels = np.loadtxt(labels_path, delimiter=',', dtype=np.int64)
            # Create a dictionary mapping graph_id to label
            self.labels_map = {graph_labels[i, 0]: graph_labels[i, 1] for i in range(len(graph_labels))}
        else:
            # For test set, we load the graph IDs directly
            test_ids_file = 'test_graph_labels.txt'
            test_ids_path = os.path.join(root, test_ids_file)
            # Test file might have just one column
            try:
                self.test_ids = np.loadtxt(test_ids_path, dtype=np.int64)
                if len(self.test_ids.shape) == 1:
                    # If it's just a 1D array
                    self.unique_graph_ids = self.test_ids
                else:
                    # If it's a 2D array
                    self.unique_graph_ids = self.test_ids[:, 0]
            except Exception as e:
                print(f"Error loading test IDs: {e}")
                # Fallback to using graph IDs from indicator file
                self.unique_graph_ids = np.unique(self.graph_ids)
                
            self.labels_map = None
    
    @property
    def processed_file_names(self):
        return []
    
    @property
    def raw_file_names(self):
        return []
    
    def len(self):
        return len(self.unique_graph_ids)
    
    def get(self, idx):
        graph_id = self.unique_graph_ids[idx]
        
        # Find the nodes for this graph
        node_mask = self.graph_ids == graph_id
        
        # Create a mapping from global node IDs to local node IDs for this graph
        node_mapping = {int(global_id): local_id for local_id, global_id in enumerate(self.node_indices[node_mask])}
        
        # Extract edges for this graph
        edge_mask = np.isin(self.edges[:, 0], self.node_indices[node_mask]) & np.isin(self.edges[:, 1], self.node_indices[node_mask])
        graph_edges = self.edges[edge_mask]
        
        # Map global node IDs to local node IDs
        edge_index = np.zeros((2, len(graph_edges)), dtype=np.int64)
        for i, (src, dst) in enumerate(graph_edges):
            edge_index[0, i] = node_mapping[int(src)]
            edge_index[1, i] = node_mapping[int(dst)]
        
        # Create node features (using node labels as features)
        # Keep node labels as integers for embedding lookup
        node_features = self.node_labels[node_mask].reshape(-1, 1).astype(np.int64)
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),  # Changed to long for embedding lookup
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        
        # Add graph label
        if self.mode == 'train' and graph_id in self.labels_map:
            data.y = torch.tensor([self.labels_map[graph_id]], dtype=torch.long)
        else:
            # Add a dummy label for test data or missing labels to avoid KeyError in batching
            data.y = torch.tensor([0], dtype=torch.long)  # Use 0 as a placeholder
        
        # Store the graph_id for test set predictions
        data.graph_id = torch.tensor([graph_id], dtype=torch.long)
        
        return data

