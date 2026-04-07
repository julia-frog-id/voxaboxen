
import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import soundfile as sf
import librosa
from scipy.io import wavfile


def load_raven_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        df = None

    if df is None or df.shape[1] <= 1:
        df = pd.read_csv(path, sep=",", engine="python")

    df.columns = [c.strip() for c in df.columns]
    return df


def load_audio_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    try:
        y, sr = sf.read(wav_path, always_2d=False)
        y = np.asarray(y)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        else:
            y = y.astype(np.float32)
        return y, int(sr)
    except Exception:
        pass

    try:
        sr, y = wavfile.read(wav_path)
        y = np.asarray(y)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        else:
            y = y.astype(np.float32)
        return y, int(sr)
    except Exception as e:
        raise RuntimeError(
            "Failed to load WAV. Install 'soundfile' (recommended) or ensure scipy can read this WAV."
        ) from e


def extract_intervals(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def get(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    begin_col = get("Begin Time (s)", "Begin Time", "Begin")
    end_col   = get("End Time (s)", "End Time", "End")

    if begin_col is None or end_col is None:
        raise ValueError(
            f"Missing required columns. Found columns: {list(df.columns)}. "
            'Need at least "Begin Time (s)" and "End Time (s)".'
        )

    out = pd.DataFrame({
        "begin_s": pd.to_numeric(df[begin_col], errors="coerce"),
        "end_s":   pd.to_numeric(df[end_col], errors="coerce"),
    }).dropna()

    out = out[out["end_s"] > out["begin_s"]].copy()
    return out


def add_outline_boxes_lines(ax, intervals, f_min, f_max, color, lw=2.0, zorder=10):
    for _, r in intervals.iterrows():
        x0 = float(r["begin_s"])
        x1 = float(r["end_s"])
        if x1 <= x0:
            continue

        ax.plot([x0, x0], [f_min, f_max], color=color, linewidth=lw, zorder=zorder, solid_capstyle="butt")
        ax.plot([x1, x1], [f_min, f_max], color=color, linewidth=lw, zorder=zorder, solid_capstyle="butt")
        ax.plot([x0, x1], [f_min, f_min], color=color, linewidth=lw, zorder=zorder, solid_capstyle="butt")
        ax.plot([x0, x1], [f_max, f_max], color=color, linewidth=lw, zorder=zorder, solid_capstyle="butt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)

    # Spectrogram params
    p.add_argument("--n_fft", type=int, default=4096, help="FFT size (bigger = better freq resolution)")
    p.add_argument("--hop", type=int, default=128, help="Hop length")
    p.add_argument("--fmin", type=float, default=200.0, help="Min frequency to display (Hz)")
    p.add_argument("--fmax", type=float, default=12000.0, help="Max frequency to display (Hz)")
    p.add_argument("--top_db", type=float, default=80.0, help="dB range to display (smaller = higher contrast)")
    p.add_argument("--no_log_freq", action="store_true", help="Use linear frequency axis (default is log)")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    # --- Load ---
    y, sr = load_audio_mono(args.wav)
    gt = extract_intervals(load_raven_table(args.gt))
    pr = extract_intervals(load_raven_table(args.pred))

    # --- Spectrogram (librosa preferred for nice scaling) ---
    try:

        # Power spectrogram
        S = np.abs(librosa.stft(y=y, n_fft=args.n_fft, hop_length=args.hop, center=True)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=args.n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=args.hop)

        # Convert to dB with clamp
        S_db = librosa.power_to_db(S, ref=np.max, top_db=args.top_db)

    except Exception as e:
        raise RuntimeError(
            "This script expects librosa for the improved spectrogram scaling. "
            "Install with: pip install librosa"
        ) from e

    # Frequency crop (avoid empty top band)
    f_min = max(args.fmin, float(freqs.min()))
    f_max = min(args.fmax, float(freqs.max()))
    if f_max <= f_min:
        raise ValueError(f"Invalid frequency range: fmin={f_min}, fmax={f_max}")

    fmask = (freqs >= f_min) & (freqs <= f_max)
    S_show = S_db[fmask, :]
    freqs_show = freqs[fmask]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(50, 6))

    # For log y-scale, pcolormesh can have issues if freq includes 0; we start at fmin>=200 by default.
    mesh = ax.pcolormesh(times, freqs_show, S_show, shading="auto", cmap="viridis", zorder=1)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.005, fraction=0.02)
    cbar.set_label(f"Power (dB, top_db={args.top_db:g})")

    if not args.no_log_freq:
        ax.set_yscale("log")
        # Keep bounds correct on log axis
        ax.set_ylim([f_min, f_max])
    else:
        ax.set_ylim([f_min, f_max])

    # Overlays
    gt_color = "#2ECC71"
    pr_color = "#F39C12"
    add_outline_boxes_lines(ax, gt, f_min, f_max, color=gt_color, lw=2.0, zorder=10)
    add_outline_boxes_lines(ax, pr, f_min, f_max, color=pr_color, lw=2.0, zorder=11)

    ax.set_title("Spectrogram with Raven Selection Outlines")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    ax.legend(
        handles=[
            Line2D([0], [0], color=gt_color, lw=2.0, label="Ground truth"),
            Line2D([0], [0], color=pr_color, lw=2.0, label="Predictions"),
        ],
        loc="upper right",
        framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)