import torch
import numpy as np

# Step 1: Load the data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file]
    data = [[int(id) for id in line] for line in data]
    return data

# Step 2: Process the data and create the adjacency matrix
def create_adjacency_matrix(data):
    user_ids = set([row[0] for row in data])
    item_ids = set([item for sublist in data for item in sublist[1:]])

    num_users = max(user_ids) + 1
    num_items = max(item_ids) + 1
    total_nodes = num_users + num_items

    adjacency_matrix = torch.zeros(total_nodes, total_nodes)

    for row in data:
        user_id = row[0]
        items = row[1:]
        for item_id in items:
            # Set user to item link
            adjacency_matrix[user_id, num_users + item_id] = 1
            # Set item to user link
            adjacency_matrix[num_users + item_id, user_id] = 1

    return user_ids, list(item_ids), adjacency_matrix

# Example usage
file_path = 'data/clothing/train.txt'
data = load_data(file_path)
data = data[:2]
user_ids, item_ids, adjacency_matrix = create_adjacency_matrix(data)

print("User IDs:", user_ids)
print("Item IDs:", item_ids)
print("Adjacency Matrix:\n", adjacency_matrix)
