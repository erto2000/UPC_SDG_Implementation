import torch


# Loss function for the selection module
# user_emb: tensor of user embeddings, shape: (batch_size, embedding_dim)
# item_emb: tensor of item embeddings, shape: (batch_size, num_items, embedding_dim)
# attention_probabilities: tensor of attention probabilities, shape: (batch_size, num_items)
def selection_module_loss(user_emb, item_emb, attention_probabilities):
    weighted_sum = torch.sum(item_emb * attention_probabilities.unsqueeze(-1), dim=1)  # shape: (batch_size, embedding_dim)
    loss = torch.nn.MSELoss()(weighted_sum, user_emb)
    return loss


# Loss function for the generation module
# user_emb: tensor of user embeddings, shape: (batch_size, embedding_dim)
# item_emb: tensor of item embeddings, shape: (batch_size, num_items, embedding_dim)
# replacement_item_emb: tensor of replacement item embeddings, shape: (batch_size, num_items, embedding_dim)
# item_emb_distances: tensor of distances between item embeddings, shape: (batch_size, num_items, num_all_items)
# privacy_preference: tensor of privacy preferences, shape: (batch_size)
# mask: tensor of mask for padding, shape: (batch_size, num_items)
def generation_module_loss(user_emb, item_emb, replacement_item_emb, item_emb_distances, privacy_preference, mask=None):
    # PRIVACY LOSS
    # Maximum and minimum of distances
    min_distances = torch.min(item_emb_distances, dim=2).values  # shape: (batch_size, num_items)
    max_distances = torch.max(item_emb_distances, dim=2).values  # shape: (batch_size, num_items)

    # Take the dot product of item embeddings and replacement item embeddings
    replacement_item_distances = torch.einsum('ijk,ijk->ij', item_emb, replacement_item_emb)  # shape: (batch_size, num_items)

    # Calculate the similarity based on the distances
    similarity = (replacement_item_distances - min_distances) / (max_distances - min_distances)  # shape: (batch_size, num_items)

    privacy_loss = similarity - privacy_preference.unsqueeze(-1)  # shape: (batch_size, num_items)
    privacy_loss = torch.clamp(privacy_loss, min=0)  # Ensure non-negativity

    # Apply mask if provided
    if mask is not None:
        privacy_loss = privacy_loss * mask

    privacy_loss = privacy_loss.sum()

    # UTILITY LOSS
    # Take the dot product of user embeddings and item embeddings
    user_item_distances = torch.einsum('ijk,ijk->ij', user_emb.unsqueeze(1), item_emb)  # shape: (batch_size, num_items)

    # Apply the sigmoid function
    utility_loss = torch.sigmoid(user_item_distances)

    # Compute the negative log of the probabilities
    utility_loss = -torch.log(utility_loss + 1e-10)  # Adding a small value to avoid log(0)

    # Apply mask if provided
    if mask is not None:
        utility_loss = utility_loss * mask

    # Sum over all user-item pairs
    utility_loss = utility_loss.sum()

    return privacy_loss, utility_loss



