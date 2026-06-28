"""
ECG Dataset Preprocessing Pipeline
====================================
Applies band-pass filtering (0.5–40 Hz), min-max normalisation,
and segments signals into 3,000-sample windows for PyTorch training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from typing import Tuple


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0,
                    fs: float = 300.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1D ECG signal."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def min_max_normalise(signal: np.ndarray) -> np.ndarray:
    """Normalise a signal to the range [-1, 1]."""
    s_min, s_max = signal.min(), signal.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(signal)
    return 2.0 * (signal - s_min) / (s_max - s_min) - 1.0


def segment_signal(signal: np.ndarray, window_size: int = 3000,
                   stride: int = 3000) -> np.ndarray:
    """Segment a long signal into fixed-length windows."""
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segments.append(signal[start: start + window_size])
    return np.array(segments)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals.

    Parameters
    ----------
    signals : np.ndarray  shape (N, seq_len)
    labels  : np.ndarray  shape (N,)  — integer class indices
    apply_filter : bool   whether to apply the band-pass filter
    """

    CLASS_MAP = {'N': 0, 'A': 1, 'O': 2}

    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 apply_filter: bool = True, fs: float = 300.0):
        self.signals = []
        self.labels  = []
        for sig, lbl in zip(signals, labels):
            if apply_filter:
                sig = bandpass_filter(sig, fs=fs)
            sig = min_max_normalise(sig)
            self.signals.append(sig.astype(np.float32))
            lbl_int = self.CLASS_MAP[lbl] if isinstance(lbl, str) else int(lbl)
            self.labels.append(lbl_int)
        self.signals = np.array(self.signals)
        self.labels  = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.signals[idx]).unsqueeze(0)   # (1, seq_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
