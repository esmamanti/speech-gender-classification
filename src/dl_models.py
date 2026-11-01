"""Derin öğrenme modelleri (PyTorch): MLP, CNN, CRNN.
Hepsi tek bir çıktı (logit) verir: sigmoid(logit) = P(kadın)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Akustik özellikler (F0 + formant) -> sinir ağı.
    Girdiler, eğitim verisinin ortalama/std'si ile modelin İÇİNDE normalize edilir
    (set_normalization). Böylece kaydedilen model tahminde de aynı ölçeği kullanır."""

    def __init__(self, n_in: int, hidden=(64, 32), dropout=0.3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        self.mean.copy_(X.mean(0))
        self.std.copy_(X.std(0).clamp_min(1e-6))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(1)


def conv_block(c_in, c_out, pool):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CNN(nn.Module):
    """Log-mel spektrogram (1 x 64 x T) -> 4 konvolüsyon bloğu -> global ortalama -> çıktı.
    Spektrogramı bir 'resim' gibi işler: frekans desenlerini (perde, formant) öğrenir."""

    def __init__(self, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        blocks, c = [], 1
        for ch in channels:
            blocks.append(conv_block(c, ch, pool=2))
            c = ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x):                      # x: (B, 1, n_mels, T)
        x = self.features(x)                   # (B, C, F', T')
        x = x.mean(dim=(2, 3))                 # global average pooling -> (B, C)
        return self.head(x).squeeze(1)


class CRNN(nn.Module):
    """CNN + GRU: CNN her zaman adımındaki frekans desenini çıkarır,
    çift yönlü GRU bu desenlerin ZAMAN içindeki değişimini (tonlama vb.) öğrenir."""

    def __init__(self, n_mels=64, channels=(16, 32, 64), rnn_hidden=64, dropout=0.3):
        super().__init__()
        blocks, c = [], 1
        for ch in channels:
            blocks.append(conv_block(c, ch, pool=2))
            c = ch
        self.features = nn.Sequential(*blocks)
        f_out = n_mels // (2 ** len(channels))          # 64 -> 8
        self.rnn = nn.GRU(input_size=c * f_out, hidden_size=rnn_hidden,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, 1))

    def forward(self, x):                      # x: (B, 1, n_mels, T)
        x = self.features(x)                   # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B, T', C*F')
        x, _ = self.rnn(x)                     # (B, T', 2*hidden)
        x = x.mean(dim=1)                      # zaman boyunca ortalama
        return self.head(x).squeeze(1)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)