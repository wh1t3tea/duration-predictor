import os
from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import Dataset


class PhoneDataset(Dataset):
    def __init__(self, root_dir, filenames):
        super().__init__()

        self.vocab = None
        self.filenames = filenames
        self.data = self.load_data(root_dir)
        self.pad_idx = self.data[0]["pad_idx"]
        self.vocab.append(self.pad_idx)
        self.vocab = sorted(self.vocab)
        self.vocab = {self.vocab[i]: i for i in range(len(self.vocab))}
        self.length = self.max_len()
        self.label2emb = {}

        for sample in self.data:
            if sample["speaker"] not in self.label2emb:
                self.label2emb[sample["speaker"]] = sample["speaker_embeddings"]

    def load_data(self, root_dir):

        voc = []
        all_data = []

        for filename in self.filenames:
            path_f = os.path.join(root_dir, filename)
            data = pd.read_pickle(path_f)
            tokens = [data[x]["tokenized_phones"] for x in range(len(data))]

            for sample in tokens:
                for token in sample:
                    voc.append(token)
            all_data += data

        voc = set(voc)

        self.vocab = list(voc)

        return all_data

    def __len__(self):
        return len(self.data)

    def max_len(self):
        max_len = max([len(self.data[x]["tokenized_phones"]) for x in range(len(self.data))])
        return max_len

    def __getitem__(self, idx):
        sample = deepcopy(self.data[idx])
        tokens, durs, speaker_embed = (
               [self.vocab[x] for x in sample["tokenized_phones"]],
               sample["durations"],
               self.label2emb[sample["speaker"]]
        )
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