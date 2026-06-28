"""
ECG GAN Training Script
========================
Trains the Generator and Discriminator on ECG signals.

Usage
-----
    python train.py --epochs 100 --batch_size 64 --latent_dim 100 --class_label N
"""
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from src.dataset import ECGDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    signals = np.load("data/ecg_signals.npy")
    labels  = np.load("data/ecg_labels.npy")
    dataset = ECGDataset(signals, labels, apply_filter=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator(latent_dim=args.latent_dim, seq_len=3000).to(device)
    D = Discriminator(seq_len=3000).to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_g, epoch_d = [], []
        for real_signals, _ in loader:
            real_signals = real_signals.to(device)
            bs = real_signals.size(0)
            real_labels = torch.ones(bs, 1).to(device)
            fake_labels = torch.zeros(bs, 1).to(device)

            # Train Discriminator
            opt_D.zero_grad()
            d_real = D(real_signals)
            d_real_loss = criterion(d_real, real_labels)
            z = torch.randn(bs, args.latent_dim).to(device)
            fake_signals = G(z).detach()
            d_fake = D(fake_signals)
            d_fake_loss = criterion(d_fake, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward(); opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(bs, args.latent_dim).to(device)
            fake_signals = G(z)
            g_loss = criterion(D(fake_signals), real_labels)
            g_loss.backward(); opt_G.step()

            epoch_g.append(g_loss.item())
            epoch_d.append(d_loss.item())

        g_losses.append(np.mean(epoch_g))
        d_losses.append(np.mean(epoch_d))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch:3d}/{args.epochs}] | G_loss={g_losses[-1]:.4f} | D_loss={d_losses[-1]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    torch.save(G.state_dict(), "results/generator.pth")
    torch.save(D.state_dict(), "results/discriminator.pth")
    np.save("results/g_losses.npy", np.array(g_losses))
    np.save("results/d_losses.npy", np.array(d_losses))
    logger.info("Training complete. Models saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--latent_dim",  type=int, default=100)
    parser.add_argument("--class_label", type=str, default="N", choices=["N","A","O"])
    args = parser.parse_args()
    train(args)
