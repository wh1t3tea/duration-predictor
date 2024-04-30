import os
from copy import deepcopy

import pandas as pd
import torch
from torch import nn


class PhoneDataset(nn.Module):
    def __init__(self, root_dir):
        super().__init__()
        self.vocab = None
        self.data = self.load_data(root_dir)
        self.pad_idx = [self.data[0]["pad_idx"]]
        self.length = self.max_len()

    def load_data(self, root_dir):
        data = sum([pd.read_pickle(filename) for filename in os.listdir(root_dir)])
        vocab = [data[x]["tokenized_phones"] for x in range(len(data))]
        voc = []
        for sample in vocab:
            for token in sample:
                voc.append(token)
        voc = set(voc)
        voc.add(29)
        self.vocab = list(voc)
        return data

    def __len__(self):
        return len(self.data)

    def max_len(self):
        max_len = max([len(self.data[x]["tokenized_phones"]) for x in range(len(self.data))])
        return max_len

    def __getitem__(self, idx):
        sample = deepcopy(self.data[idx])
        tokens, durs, speaker_embed = [self.voc_[x] for x in sample["tokenized_phones"]], sample["durations"], sample["speaker_embeddings"]
        mask = [0] * (self.length - len(tokens))
        tokens += mask
        durs += mask
        dur_mask = [True] * (self.length - len(mask))
        dur_mask += [False] * (self.length - len(dur_mask))
        speaker_embed = torch.tensor(speaker_embed[0])
        return {"tokens": torch.tensor(tokens).int(),
                "mask": torch.tensor(dur_mask),
                "lengths": torch.tensor(durs).int(),
                "sp_e": speaker_embed}