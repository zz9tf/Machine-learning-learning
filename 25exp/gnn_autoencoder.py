import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GraphConv, 
    GCNConv,
    global_mean_pool,
    global_add_pool
)

class GNNAutoencoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_labels=10, output_dim=7, embedding_dim=3):
        super(GNNAutoencoder, self).__init__()
        
        # Node label embedding
        self.node_embedding = torch.nn.Embedding(num_node_labels, embedding_dim)
        
        # capture features from node labels
        self.conv1 = GraphConv(embedding_dim, hidden_channels)  # Input is now embedding dimension
        self.conv2 = GraphConv(hidden_channels, hidden_channels*16)
        # self.conv3 = GraphConv(hidden_channels*16, hidden_channels*32)
        
        # force to capture common features
        # Encoder layers
        self.enc_conv1 = GraphConv(hidden_channels*16, hidden_channels*4)
        
        # Decoder layers
        self.dec_conv1 = GraphConv(hidden_channels*4, hidden_channels*16)
        
        # Output layer to reconstruct node labels
        self.lin = torch.nn.Linear(hidden_channels*4, output_dim)

    def forward(self, x, edge_index, batch):
        # Encoding
        node_labels = x.long().squeeze(-1)
        x = self.node_embedding(node_labels)
        
        x = F.leaky_relu(self.conv1(x, edge_index))
        original_node_features = F.leaky_relu(self.conv2(x, edge_index))
        # original_node_features = F.leaky_relu(self.conv3(x, edge_index))
        
        # Encoder - 获取节点级别的编码特征
        latent_node_features = F.leaky_relu(self.enc_conv1(original_node_features, edge_index))
        
        # 对节点特征进行池化得到图级特征，用于分类任务
        pooled_features = global_mean_pool(latent_node_features, batch)
        graph_logits = self.lin(pooled_features)
        
        # Decoder - 从节点级别的latent特征开始解码，而不是从池化后的特征
        reconstructed_features = F.leaky_relu(self.dec_conv1(latent_node_features, edge_index))
        
        return {
            'graph_logits': graph_logits,  # 图分类输出
            'reconstructed_features': reconstructed_features,  # 重建的节点特征
            'original_node_features': original_node_features  # 原始节点特征
        }
    
    def loss_function(self, outputs, y):
        # 图分类损失
        classification_loss = F.cross_entropy(outputs['graph_logits'], y)
        
        # 重建损失 - 计算原始节点特征和重建节点特征之间的MSE
        reconstruction_loss = F.mse_loss(outputs['reconstructed_features'], outputs['original_node_features'])
        
        # 总损失 - 可以使用权重调整两个损失的重要性
        total_loss = classification_loss + 1.5*reconstruction_loss
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'reconstruction_loss': reconstruction_loss
        } 