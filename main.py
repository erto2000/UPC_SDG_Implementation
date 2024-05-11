from DataLoader import load_data, get_user_item_interaction, UserItemDataset, collate_fn
from UPC_SDG import upc_sdg
import LossFunctions
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Constants
skip_training = False
saved_model_path = 'models/upc_sdg.pth'
dataset_path = 'data/clothing_data.txt'
embeddings_path = 'data/clothing_embeddings_64.pth.tar'
replacement_ratio = 0.5
privacy_preference = 0.5
num_epochs = 5
batch_size = 32

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load user and item embeddings
embeddings = torch.load(embeddings_path)
user_embeddings = embeddings['embedding_user.weight']
item_embeddings = embeddings['embedding_item.weight']

# Load training data
data = load_data(dataset_path)

# Get user and item interaction data
user_ids, interacted_item_ids, user_id_count, item_id_count = get_user_item_interaction(data)
print("Number of users:", user_id_count)
print("Number of items:", item_id_count)

# Create DataLoader
dataset = UserItemDataset(user_ids, interacted_item_ids, [privacy_preference] * len(user_ids))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = upc_sdg(user_embeddings, item_embeddings).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Check if model exists and load
if os.path.isfile(saved_model_path):
    print("Loading saved model...")
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully.")
else:
    skip_training = False
    print("No saved model found. Training from scratch.")

# Train the model
if not skip_training:
    # Iterate over each epoch
    for epoch in range(num_epochs):
        # Wrap data loader with tqdm for a progress bar
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

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
            user_emb, item_emb, item_emb_distances, attention_probabilities, replacement_item_embeddings, replacement_item_indices = model(user_ids_batch, items_ids_batch, privacy_preferences, mask)

            # Calculate the loss
            selection_module_loss = LossFunctions.selection_module_loss(user_emb, item_emb, attention_probabilities)
            # generation_module_privacy_loss, generation_module_utility_loss = LossFunctions.generation_module_loss(user_emb, item_emb, replacement_item_embeddings, item_emb_distances, privacy_preferences, mask)
            # loss = selection_module_loss + generation_module_privacy_loss + generation_module_utility_loss
            loss = selection_module_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate the average loss for the epoch
        progress_bar.close()
        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

    # If save file already exists, add number to the end
    if os.path.isfile(saved_model_path):
        i = 1
        while os.path.isfile(saved_model_path):
            saved_model_path = saved_model_path.replace('.pth', f'_{i}.pth')
            i += 1

    # If directory does not exist, create it
    if not os.path.exists(os.path.dirname(saved_model_path)):
        os.makedirs(os.path.dirname(saved_model_path))

    # Save the model after training is complete
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, saved_model_path)
    print(f"Model saved to {saved_model_path}")


# Test the model
# Wrap data loader with tqdm for a progress bar
progress_bar = tqdm(data_loader, desc=f'Testing all data', leave=True)

# Iterate over each batch
total_accuracy = 0
for user_ids_batch, items_ids_batch, privacy_preferences, mask in progress_bar:
    # Move data to device
    user_ids_batch = user_ids_batch.to(device)
    items_ids_batch = items_ids_batch.to(device)
    privacy_preferences = privacy_preferences.to(device)
    mask = mask.to(device)

    # Forward pass
    _, _, _, attention_probabilities, _, replacement_item_indices = model(user_ids_batch, items_ids_batch, privacy_preferences, mask)

    # Number of items to replace based on the replacement ratio
    item_counts = mask.sum(dim=1) # shape: (batch_size)
    number_of_replacements = (item_counts * replacement_ratio).int()
    number_of_replacements[number_of_replacements == 0] = 1

    # Sort the item indices based on the attention probabilities
    masked_attention_probabilities = torch.where(mask, attention_probabilities, float('inf'))  # shape: (batch_size, num_items)
    _, sorted_indices = torch.sort(masked_attention_probabilities, dim=1)  # shape: (batch_size, num_items)

    # Sort items_ids_batch and replacement_item_indices based on the sorted indices
    sorted_item_indices = torch.gather(items_ids_batch, 1, sorted_indices)  # shape: (batch_size, num_items)
    sorted_replacement_item_indices = torch.gather(replacement_item_indices, 1, sorted_indices)  # shape: (batch_size, num_items)

    # Generate mask for replacement
    batch_size, num_items = sorted_item_indices.shape
    indices_range = torch.arange(num_items).unsqueeze(0).repeat(batch_size, 1).to(device)  # shape: (batch_size, num_items)
    replacement_mask = indices_range < number_of_replacements.unsqueeze(1)

    # Update sorted_item_emb with values from sorted_replacement_item_embeddings where replacement_mask is True
    sorted_item_indices[replacement_mask] = sorted_replacement_item_indices[replacement_mask]

    # Forward pass again with the updated item indices
    _, item_emb, _, attention_probabilities, _, _ = model(user_ids_batch, sorted_item_indices, privacy_preferences, mask)

    # Calculate feature vector based on attention probabilities
    weighted_sum = torch.sum(item_emb * attention_probabilities.unsqueeze(-1), dim=1)  # shape: (batch_size, embedding_dim)

    # Get the closest user id to the feature vector
    closest_user_id = model.get_closest_user_id(weighted_sum)

    # Check if the closest user id is the same as the original user id
    correct = closest_user_id == user_ids_batch

    # Calculate accuracy
    accuracy = correct.sum().item() / batch_size
    total_accuracy += accuracy

print(f"Average Accuracy: {total_accuracy / len(data_loader)}")



















