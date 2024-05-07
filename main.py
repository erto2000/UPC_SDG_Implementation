from DataLoader import load_data, get_user_item_interaction, UserItemDataset, collate_fn
from SecureLightGCN import SecureLightGCN
import LossFunctions
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Constants
train_test_ratio = 0.9
embedding_dim = 32
replacement_ratio = 0.1
privacy_preference = 0.5
num_epochs = 100
batch_size = 64

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
train_file_path = 'data/clothing/data.txt'
data = load_data(train_file_path)

# Get user and item interaction data
user_ids, interacted_item_ids, user_id_count, item_id_count = get_user_item_interaction(data)
print("Number of users:", user_id_count)
print("Number of items:", item_id_count)

# Create DataLoader
dataset = UserItemDataset(user_ids, interacted_item_ids, [privacy_preference] * len(user_ids))
train_count = int(train_test_ratio * len(dataset))
test_count = len(dataset) - train_count
train_dataset, test_dataset = random_split(dataset, [train_count, test_count])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model
model = SecureLightGCN(user_id_count, item_id_count, embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate over each epoch
for epoch in range(num_epochs):
    # Wrap data loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

    # Iterate over each batch
    total_loss = 0
    for user_ids_batch, items_ids_batch, privacy_preferences, mask in progress_bar:
        # Move data to device
        user_ids_batch = user_ids_batch.to(device)
        items_ids_batch = items_ids_batch.to(device)
        privacy_preferences = privacy_preferences.to(device)
        mask = mask.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        user_emb, item_emb, item_emb_distances, attention_probabilities, replacement_item_embeddings = model(user_ids_batch, items_ids_batch, privacy_preferences, mask)

        # Calculate the loss
        selection_module_loss = LossFunctions.selection_module_loss(user_emb, item_emb, attention_probabilities)
        generation_module_privacy_loss, generation_module_utility_loss = LossFunctions.generation_module_loss(user_emb, item_emb, replacement_item_embeddings, item_emb_distances, privacy_preferences, mask)
        loss = 20000 * selection_module_loss + 50 * generation_module_privacy_loss + generation_module_utility_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item()

        # Update the progress bar with the current loss
        progress_bar.set_postfix({'loss': loss.item()})

    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)
    progress_bar.close()
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")
