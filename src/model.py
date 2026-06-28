"""
GAN Architecture for ECG Signal Synthesis
==========================================
Defines the Generator and Discriminator architectures used to synthesise
realistic physiological ECG waveforms for Normal (N), Arrhythmia (A),
and Other (O) classes.

Architecture
------------
Generator  : Latent vector (100-dim) → FC layers → Tanh → (1, 3000) waveform
Discriminator: (1, 3000) waveform → FC layers → Sigmoid → real/fake probability
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Fully-connected generator that maps a latent noise vector to a 3,000-sample
    ECG waveform normalised to [-1, 1].

    Parameters
    ----------
    latent_dim : int   Dimensionality of the input noise vector (default 100)
    seq_len    : int   Length of the output ECG sequence (default 3000)
    """

    def __init__(self, latent_dim: int = 100, seq_len: int = 3000):
        super(Generator, self).__init__()
        self.seq_len = seq_len

        def _block(in_feat: int, out_feat: int, normalise: bool = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalise:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *_block(latent_dim, 256, normalise=False),
            *_block(256, 512),
            *_block(512, 1024),
            *_block(1024, 2048),
            nn.Linear(2048, seq_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor of shape (batch_size, latent_dim)

        Returns
        -------
        Tensor of shape (batch_size, 1, seq_len)
        """
        out = self.model(z)
        return out.view(out.size(0), 1, self.seq_len)


class Discriminator(nn.Module):
    """
    Fully-connected discriminator that classifies a 3,000-sample ECG waveform
    as real or synthetic.

    Parameters
    ----------
    seq_len : int   Length of the input ECG sequence (default 3000)
    """

    def __init__(self, seq_len: int = 3000):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(seq_len, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        signal : Tensor of shape (batch_size, 1, seq_len)

        Returns
        -------
        Tensor of shape (batch_size, 1) — probability of being real
        """
        flat = signal.view(signal.size(0), -1)
        return self.model(flat)


class ConditionalGenerator(nn.Module):
    """
    Conditional GAN generator that conditions on a class label (N=0, A=1, O=2)
    to generate class-specific ECG waveforms.
    """

    def __init__(self, latent_dim: int = 100, n_classes: int = 3, seq_len: int = 3000):
        super(ConditionalGenerator, self).__init__()
        self.seq_len = seq_len
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, seq_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_input = self.label_emb(labels)
        gen_input   = torch.cat((z, label_input), dim=-1)
        out = self.model(gen_input)
        return out.view(out.size(0), 1, self.seq_len)
