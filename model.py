from torch import nn
from modules import Encoder, Decoder


class DurationPredictor(nn.Module):
    def __init__(self,
                 voc_size,
                 in_channels=256,
                 filter_channels=256,
                 hidden_dim=256,
                 speaker_emb_size=2048,
                 n_convs=2):
        super().__init__()

        self.encoder = Encoder(speaker_emb_size,
                               in_channels,
                               filter_channels,
                               n_convs,
                               hidden_dim,
                               voc_size)

        self.decoder = Decoder(hidden_dim // 2,
                               filter_channels)

    def forward(self, x, y, mask):
        x = self.encoder(x, y, mask)
        generated = self.decoder(x)
        return generated
