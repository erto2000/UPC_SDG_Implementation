import torch
import torch.nn as nn
import torch.nn.functional as F


class upc_sdg(nn.Module):
    def __init__(self, user_embeddings, item_embeddings):
        super(upc_sdg, self).__init__()

        # User and item embeddings
        self.user_embedding = user_embeddings
        self.item_embedding = item_embeddings

        embedding_dim = user_embeddings.shape[1]

        self.selection_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, 1),
            nn.LeakyReLU()
        )

        self.feature_transform = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_dim),
            # torch.nn.Linear(embedding_dim, embedding_dim)
        )

        self.generation_layer = nn.Linear((2 * embedding_dim) + 1, embedding_dim)

    # Forward pass
    # user_indices: tensor of user indices, shape: (batch_size)
    # interacted_item_indices: tensor of interacted item indices, shape: (batch_size, num_items)
    # privacy_preferences: tensor of privacy preferences, shape: (batch_size)
    # mask: tensor of mask for padding, shape: (batch_size, num_items)
    # Returns:
    # user_emb, shape: (batch_size, embedding_dim)
    # item_emb, shape: (batch_size, num_items, embedding_dim)
    # attention_probabilities, shape: (batch_size, num_items)
    # replacement_item_indices, shape: (batch_size, num_items)
    # replacement_item_embeddings, shape: (batch_size, num_items, embedding_dim)
    def forward(self, user_indices, interacted_item_indices, privacy_preferences, mask=None):
        # SELECTION MODULE
        # Get initial embeddings
        user_emb = self.user_embedding[user_indices]  # shape: (batch_size, embedding_dim)
        item_emb = self.item_embedding[interacted_item_indices]  # shape: (batch_size, num_items, embedding_dim)

        # Combine user_emb with each item_emb
        user_emb_expanded = user_emb.unsqueeze(1)  # Add 1 to the middle for broadcasting, shape: (batch_size, 1, embedding_dim)
        combined_emb = torch.cat((user_emb_expanded.expand_as(item_emb), item_emb), dim=2)  # shape: (batch_size, num_items, 2*embedding_dim)

        # Pass the combined tensor through the sequential model
        attention = self.selection_layer(combined_emb).squeeze(-1)  # Remove the last dimension, shape: (batch_size, num_items)

        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))  # Replace padding attention with negative infinity

        # Subtract max for numerical stability before softmax
        attention_probabilities = F.softmax(attention, dim=1)  # Apply softmax across items within the batch


        # GENERATION MODULE
        # Combine the embeddings with the privacy preferences
        batch_size, num_items, _ = combined_emb.shape
        privacy_preference_tensor = privacy_preferences.unsqueeze(1).unsqueeze(2).expand(batch_size, num_items, 1)  # shape: (batch_size, num_items, 1)
        privacy_combined_emb = torch.cat((combined_emb, privacy_preference_tensor), dim=2)  # shape: (batch_size, num_items, 2*embedding_dim + 1)

        # Generate latent feature
        latent_feature = self.generation_layer(privacy_combined_emb)  # shape: (batch_size, num_items, embedding_dim)
        similarity_score = torch.matmul(latent_feature, self.item_embedding.T)  # shape: (batch_size, num_items, num_all_items)

        # Suppress the similarity score of interacted items
        item_mask = torch.zeros_like(similarity_score)
        for batch_idx, (indices, indices_mask) in enumerate(zip(interacted_item_indices, mask)):
            indices = indices[indices_mask]
            item_mask[batch_idx, :, indices] = -float('inf')
        similarity_score = similarity_score + item_mask

        # Most similar item selection (Item generation)
        hard_similarity_score = F.gumbel_softmax(similarity_score, dim=-1, hard=True)  # shape: (batch_size, num_items, num_all_items)
        replacement_item_embeddings = torch.matmul(hard_similarity_score, self.item_embedding)  # shape: (batch_size, num_items, embedding_dim)

        # Get the id of the most similar item
        _, replacement_item_indices = torch.max(hard_similarity_score, dim=-1)  # shape: (batch_size, num_items)

        return user_emb, item_emb, attention_probabilities, replacement_item_indices, replacement_item_embeddings


    # Get the closest user id to the given embedding
    # embedding: tensor of embeddings, shape: (batch_size, embedding_dim)
    def get_closest_user_id(self, embedding):
        embedding = self.feature_transform(embedding)
        user_distances = torch.matmul(embedding, self.user_embedding.T)  # shape: (batch_size, num_items)
        _, closest_user_id = torch.max(user_distances, dim=1)
        return closest_user_id


    # Loss function for the selection module
    # user_emb: tensor of user embeddings, shape: (batch_size, embedding_dim)
    # item_emb: tensor of item embeddings, shape: (batch_size, num_items, embedding_dim)
    # attention_probabilities: tensor of attention probabilities, shape: (batch_size, num_items)
    def selection_module_loss(self, user_emb, item_emb, attention_probabilities):
        weighted_sum = torch.sum(item_emb * attention_probabilities.unsqueeze(-1), dim=1)  # shape: (batch_size, embedding_dim)
        weighted_sum = self.feature_transform(weighted_sum)
        loss = torch.nn.MSELoss()(weighted_sum, user_emb)
        return loss


    # Loss function for the generation module
    # user_emb: tensor of user embeddings, shape: (batch_size, embedding_dim)
    # item_emb: tensor of item embeddings, shape: (batch_size, num_items, embedding_dim)
    # replacement_item_emb: tensor of replacement item embeddings, shape: (batch_size, num_items, embedding_dim)
    # privacy_preference: tensor of privacy preferences, shape: (batch_size)
    # mask: tensor of mask for padding, shape: (batch_size, num_items)
    def generation_module_loss(self, user_emb, item_emb, replacement_item_emb, privacy_preference, mask=None):
        # PRIVACY LOSS
        # Calculate item distances (cosine similarity of replacement_item_embeddings with every other item embeddings)
        replacement_item_emb_distances = torch.matmul(replacement_item_emb, self.item_embedding.T)  # shape: (batch_size, num_items, num_all_items)

        # Maximum and minimum of distances
        min_distance = torch.min(replacement_item_emb_distances, dim=2).values  # shape: (batch_size, num_items)
        max_distance = torch.max(replacement_item_emb_distances, dim=2).values  # shape: (batch_size, num_items)

        # Take the dot product of item embeddings and replacement item embeddings
        item_emb_distance = torch.einsum('ijk,ijk->ij', item_emb, replacement_item_emb)  # shape: (batch_size, num_items)

        # Calculate the similarity based on the distances
        similarity = (item_emb_distance - min_distance) / (max_distance - min_distance)  # shape: (batch_size, num_items)

        privacy_loss = similarity - privacy_preference.unsqueeze(-1)  # shape: (batch_size, num_items)
        privacy_loss = torch.clamp(privacy_loss, min=0)  # Ensure non-negativity

        # Apply mask if provided
        if mask is not None:
            privacy_loss = privacy_loss * mask

        privacy_loss = privacy_loss.mean()

        # UTILITY LOSS
        # Take the dot product of user embeddings and item embeddings
        user_item_distances = torch.einsum('ijk,ijk->ij', user_emb.unsqueeze(1), replacement_item_emb)  # shape: (batch_size, num_items)

        # Apply the sigmoid function
        utility_loss = torch.sigmoid(user_item_distances)

        # Compute the negative log of the probabilities
        utility_loss = -torch.log(utility_loss + 1e-10)  # Adding a small value to avoid log(0)

        # Apply mask if provided
        if mask is not None:
            utility_loss = utility_loss * mask

        # Sum over all user-item pairs
        utility_loss = utility_loss.mean()

        return privacy_loss, utility_loss
