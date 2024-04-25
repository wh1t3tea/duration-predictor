from copy import deepcopy

import torch
from torch import nn


class PhoneDataset(nn.Module):
    def __init__(self, data, voc_dict):
        super().__init__()
        self.data = data
        self.pad_idx = [self.data[0]["pad_idx"]]
        self.length = self.max_len()
        self.voc_ = voc_dict

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
                "sp_embd": speaker_embed}