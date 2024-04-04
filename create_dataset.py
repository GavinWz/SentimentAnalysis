import json
import pickle

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SentimentDataset(Dataset):
    def __init__(self, file_path, pad_token_id=0):
        super(SentimentDataset, self).__init__()
        self.PAD_ID = pad_token_id
        with open(file_path, 'rb') as reader:
            self.data = pickle.load(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review_ids, label = self.data[idx]
        return review_ids, label, len(review_ids)

    def collate_fn(self, batch):
        review_ids, label, lengths = list(zip(*batch))
        max_len = max(lengths)
        mask = np.zeros(shape=(len(review_ids), max_len), dtype=np.float32)
        for i in range(len(review_ids)):
            review_ids[i].extend([self.PAD_ID] * (max_len - lengths[i]))
            mask[i][:lengths[i]] = 1
        x = torch.tensor(review_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        mask = torch.from_numpy(mask)
        return x, y, mask

def create_dataloader(file_path, batch_size, shuffle=True, num_workers=0):
    dataset = SentimentDataset(file_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )
    return dataloader

if __name__ == '__main__':
    tokens = json.load(open('datas/tokens.json', 'r', encoding='utf-8'))
    data = create_dataloader('datas/train.pkl', batch_size=4)
    i = 0
    for x, y, mask in data:
        i += 1
        print(x)
        print(y)
        print(len(x))
        print(mask)
        x_text = [' '.join(tokens[token_id] for token_id in token_ids if token_id > 0)
                  for token_ids in x.detach().numpy()]
        print(x_text)
        if i == 3:
            break