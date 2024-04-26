import torch
from SecureLightGCN import SecureLightGCN

# Step 1: Load the data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file]
    data = [[int(id) for id in line] for line in data]
    return data


# Example usage
file_path = 'data/clothing/train.txt'
data = load_data(file_path)
data = data[:2]

user_ids = [row[0] for row in data]
interacted_item_ids = [row[1:] for row in data]
user_id_count = max(user_ids) + 1
item_id_count = max(max(subarray) for subarray in interacted_item_ids) + 1

print("Number of users:", user_id_count)
print("Number of items:", item_id_count)

embedding_dim = 1
n_layers = 1

model = SecureLightGCN(user_id_count, item_id_count, embedding_dim, n_layers)

# Get predicted scores
scores = model(torch.tensor(user_ids[0]), torch.tensor(interacted_item_ids[0]))
print(scores)