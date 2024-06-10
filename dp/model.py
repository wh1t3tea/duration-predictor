from torch import nn
import torch
from modules import Encoder, Decoder


class DurationPredictor(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_convs,
                 max_seq_len,
                 in_channels=256,
                 filter_channels=256,
                 hidden_dim=256,
                 speaker_emb_size=2048):
        super().__init__()

        self.encoder = Encoder(
            speaker_emb_size=speaker_emb_size,
            in_channels=in_channels,
            filter_channels=filter_channels,
            n_convs=n_convs,
            hidden_dim=hidden_dim,
            voc_size=vocab_size,
            max_seq_len=max_seq_len,
        )

        self.decoder = Decoder(
            hidden_dim=hidden_dim // 2,
            filter_channels=filter_channels
        )

    def forward(self, x, y):
        x = self.encoder(x, y)
        outp = self.decoder(x)
        return outp
