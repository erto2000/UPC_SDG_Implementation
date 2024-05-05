import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class UserItemDataset(Dataset):
    def __init__(self, user_ids, interacted_item_ids, privacy_preferences):
        self.user_ids = user_ids
        self.interacted_item_ids = interacted_item_ids
        self.privacy_preferences = privacy_preferences

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        items = self.interacted_item_ids[idx]
        privacy_preference = self.privacy_preferences[idx]
        return user_id, items, privacy_preference


# Function to load data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file]
    return [[int(id) for id in line] for line in data]


def get_user_item_interaction(data):
    # Preprocess to extract user and item IDs
    user_ids = [row[0] for row in data]
    interacted_item_ids = [row[1:] for row in data]
    user_id_count = max(user_ids) + 1
    item_id_count = max(max(items) for items in interacted_item_ids if items) + 1

    return user_ids, interacted_item_ids, user_id_count, item_id_count


def collate_fn(batch):
    user_ids, item_ids, privacy_preferences = zip(*batch)
    user_tensor = torch.tensor(user_ids)
    privacy_tensor = torch.tensor(privacy_preferences)

    items_tensor = pad_sequence([torch.tensor(items) for items in item_ids], batch_first=True, padding_value=0)
    mask = [torch.ones(len(items), dtype=torch.bool) for items in item_ids]
    mask = pad_sequence(mask, batch_first=True, padding_value=False)

    return user_tensor, items_tensor, privacy_tensor, mask
