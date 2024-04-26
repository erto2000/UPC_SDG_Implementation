import torch
import torch.nn as nn
import torch.nn.functional as F


class SecureLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, n_layers):
        super(SecureLightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.select_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, 1),
            nn.LeakyReLU()
        )

        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)


    def forward(self, user_indice, interacted_item_indices):
        # Get initial embeddings
        user_emb = self.user_embedding(user_indice)  # shape: (embedding_dim,)
        item_emb = self.item_embedding(interacted_item_indices)  # shape: (num_items, embedding_dim)

        # Duplicate user_emb to match the number of item_emb
        user_emb_expanded = user_emb.unsqueeze(0).repeat(item_emb.size(0), 1)  # shape: (num_items, embedding_dim)

        # Combine user_emb with each item_emb
        combined_emb = torch.cat((user_emb_expanded, item_emb), dim=1)  # shape: (num_items, 2*embedding_dim)

        # Pass the combined tensor through the sequential model
        attention = self.select_layer(combined_emb).squeeze(-1)  # Remove the last dimension to prepare for softmax

        # Apply softmax to the outputs to generate probabilities
        attention_probabilities = F.softmax(attention, dim=0)

        print(attention_probabilities)