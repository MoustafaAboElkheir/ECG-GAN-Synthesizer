# ECG Signal Generation with GANs

A PyTorch-based deep learning project that synthesizes physiological ECG data (PhysioNet-style) using Generative Adversarial Networks (GANs).

## Overview
This repository contains the implementation of a GAN designed to generate realistic ECG signals for Normal (N), Arrhythmia (A), and Other (O) classes. The project addresses data scarcity in biomedical machine learning by providing high-quality synthetic training data.

## Key Components
- **Preprocessing Pipeline:** Applies a 0.5-40 Hz band-pass filter, min-max normalization, and segments data into 3,000-sample windows.
- **Discriminator:** A stable, multi-layer PyTorch architecture optimized for physiological time-series data.
- **Generator:** Transforms latent space vectors into realistic 3,000-sample ECG waveforms.
- **Evaluation:** Compares synthetic vs. real signals using frequency domain analysis and morphological features.

## Requirements
- PyTorch >= 2.0
- NumPy, Pandas, SciPy
- Matplotlib

## Usage
```bash
# Train the GAN
python train.py --epochs 100 --batch_size 64

# Generate synthetic samples
python generate.py --samples 1000 --output synthetic_ecg.npy
```

*Created by Moustafa AbouElkheir*
