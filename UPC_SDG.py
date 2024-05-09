import torch
import torch.nn as nn
import torch.nn.functional as F


class upc_sdg(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(upc_sdg, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.selection_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, 1),
            nn.LeakyReLU()
        )

        self.generation_layer = nn.Linear((2 * self.embedding_dim) + 1, self.embedding_dim)

        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    # Forward pass
    # user_indices: tensor of user indices, shape: (batch_size)
    # interacted_item_indices: tensor of interacted item indices, shape: (batch_size, num_items)
    # privacy_preferences: tensor of privacy preferences, shape: (batch_size)
    # mask: tensor of mask for padding, shape: (batch_size, num_items)
    # Returns:
    # user_emb, shape: (batch_size, embedding_dim)
    # item_emb, shape: (batch_size, num_items, embedding_dim)
    # item_emb_distances, shape: (batch_size, num_items, num_all_items)
    # attention_probabilities, shape: (batch_size, num_items)
    # replacement_item_embeddings, shape: (batch_size, num_items, embedding_dim)
    # replacement_item_indices, shape: (batch_size, num_items)
    def forward(self, user_indices, interacted_item_indices, privacy_preferences, mask=None):
        # SELECTION MODULE
        # Get initial embeddings
        user_emb = self.user_embedding(user_indices)  # shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding(interacted_item_indices)  # shape: (batch_size, num_items, embedding_dim)

        # Calculate item distances (cosine similarity of item_emb with every other item embeddings)
        item_emb_distances = torch.matmul(item_emb, self.item_embedding.weight.T)  # shape: (batch_size, num_items, num_all_items)

        # Combine user_emb with each item_emb
        user_emb_expanded = user_emb.unsqueeze(1)  # Add 1 to the middle for broadcasting, shape: (batch_size, 1, embedding_dim)
        combined_emb = torch.cat((user_emb_expanded.expand_as(item_emb), item_emb), dim=2)  # shape: (batch_size, num_items, 2*embedding_dim)

        # Pass the combined tensor through the sequential model
        attention = self.selection_layer(combined_emb).squeeze(-1)  # Remove the last dimension, shape: (batch_size, num_items)

        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))  # Replace padding attention with negative infinity

        # Subtract max for numerical stability before softmax
        # max_val = torch.max(attention, dim=1, keepdim=True).values  # Calculate max per batch for stability
        attention_probabilities = F.softmax(attention, dim=1)  # Apply softmax across items within the batch


        # GENERATION MODULE
        # Combine the embeddings with the privacy preferences
        batch_size, num_items, _ = combined_emb.shape
        privacy_preference_tensor = privacy_preferences.unsqueeze(1).unsqueeze(2).expand(batch_size, num_items, 1)  # shape: (batch_size, num_items, 1)
        privacy_combined_emb = torch.cat((combined_emb, privacy_preference_tensor), dim=2)  # shape: (batch_size, num_items, 2*embedding_dim + 1)

        # Generate latent feature
        latent_feature = self.generation_layer(privacy_combined_emb)  # shape: (batch_size, num_items, embedding_dim)
        similarity_score = torch.matmul(latent_feature, self.item_embedding.weight.T)  # shape: (batch_size, num_items, num_all_items)

        # Suppress the similarity score of interacted items
        item_mask = torch.zeros_like(similarity_score)
        for batch_idx, (indices, indices_mask) in enumerate(zip(interacted_item_indices, mask)):
            indices = indices[indices_mask]
            item_mask[batch_idx, :, indices] = -float('inf')
        similarity_score = similarity_score + item_mask

        # Most similar item selection (Item generation)
        hard_similarity_score = F.gumbel_softmax(similarity_score, dim=-1, hard=True)  # shape: (batch_size, num_items, num_all_items)
        replacement_item_embeddings = torch.matmul(hard_similarity_score, self.item_embedding.weight)  # shape: (batch_size, num_items, embedding_dim)

        # Get the id of the most similar item
        _, replacement_item_indices = torch.max(hard_similarity_score, dim=-1)  # shape: (batch_size, num_items)

        return user_emb, item_emb, item_emb_distances, attention_probabilities, replacement_item_embeddings, replacement_item_indices


    def get_closest_user_id(self, user_emb):
        user_item_distances = torch.matmul(user_emb, self.item_embedding.weight.T)  # shape: (batch_size, num_items)
        _, closest_user_id = torch.max(user_item_distances, dim=1)
        return closest_user_id
