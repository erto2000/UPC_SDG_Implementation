from DataLoader import load_data, get_user_item_interaction, UserItemDataset, collate_fn
from SecureLightGCN import SecureLightGCN
from LossFunctions import selection_module_loss, generation_module_loss
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# Constants
embedding_dim = 32
replacement_ratio = 0.1
privacy_preference = 0.5


# Load training data
train_file_path = 'data/clothing/data.txt'
data = load_data(train_file_path)
data = data[:10]

# Get user and item interaction data
user_ids, interacted_item_ids, user_id_count, item_id_count = get_user_item_interaction(data)
print("Number of users:", user_id_count)
print("Number of items:", item_id_count)

# Create DataLoader
dataset = UserItemDataset(user_ids, interacted_item_ids, [privacy_preference] * len(user_ids))
train_count = int(0.9 * len(dataset))
test_count = len(dataset) - train_count
train_dataset, test_dataset = random_split(dataset, [train_count, test_count])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize the model
model = SecureLightGCN(user_id_count, item_id_count, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Set the number of epochs
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    # Wrap your data loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    for user_ids_batch, items_ids_batch, privacy_preferences, mask in progress_bar:
        optimizer.zero_grad()
        user_emb, item_emb, attention_probabilities, replacement_item_embeddings, item_emb_distances = model(user_ids_batch, items_ids_batch, privacy_preferences, mask)
        selection_module_loss = selection_module_loss(user_emb, item_emb, attention_probabilities)
        generation_module_loss = generation_module_loss(user_emb, item_emb, replacement_item_embeddings, item_emb_distances, privacy_preferences, mask)
        loss = selection_module_loss + generation_module_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Optionally update the progress bar with the current loss
        progress_bar.set_postfix({'loss': loss.item()})

    average_loss = total_loss / len(train_loader)
    progress_bar.close()
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

# # Calculate hit rate or precision of top-k recommendations
# hits = 0
# total = 0
# k = 10  # Top-k items
# model.eval()  # Make sure the model is in eval mode
# with torch.no_grad():  # Disable gradient computation during inference
#     for user_ids_batch, items_ids_batch, mask in test_loader:
#         attention_scores = model(user_ids_batch, items_ids_batch, mask)  # Get model outputs in inference
#
#         # TODO: It should not look for top atentions. It should look for most similar item. Fix this
#         k = min(k, mask.sum(dim=1).min().item())  # Adjust k to the number of valid items in the batch
#         _, top_k_indices = torch.topk(attention_scores, k, dim=1, largest=True, sorted=True)
#
#         for i, user_id in enumerate(user_ids_batch):
#             actual_items = set(items_ids_batch[i][mask[i]].tolist())  # Apply mask to actual items
#             recommended_items = set(top_k_indices[i].tolist())
#
#             if actual_items.intersection(recommended_items):
#                 hits += 1
#             total += 1
#             print(actual_items)
#             print(recommended_items)
#             print()
#
# precision_at_k = hits / total
# print(f"Precision@{k}: {precision_at_k}")
