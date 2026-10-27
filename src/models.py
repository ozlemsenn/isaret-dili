"""Kelime tanıma modeli: kareleri sırayla okuyan küçük bir GRU."""
import torch.nn as nn


class KucukGRU(nn.Module):
    def __init__(self, n_feat=201, n_class=4, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * hidden, n_class))

    def forward(self, x):                 # x: (batch, kare, 201)
        h, _ = self.gru(self.inp(x))      # (batch, kare, 256)
        return self.head(h.mean(dim=1))   # (batch, sınıf sayısı)
