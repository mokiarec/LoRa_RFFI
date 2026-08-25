#!/usr/bin/env python3
"""
paper/scripts/plot_feature_visualization.py

Feature visualization across iterative distillation rounds on ResNet.
2×2 subplots: Teacher → Student R1 → Student R2 → Student R3
Each subplot shows t-SNE projection of the embedding space, colored by device label.

IEEE single-column figure: authored at final print size (3.5 in wide),
Times-compatible serif, 7–8 pt text so it renders at true size in the paper.

Output: paper/figures/feature_visualization.pdf
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# ── Path setup ─────────────────────────────────────────────────────────
_parent = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_parent))

from core.config import PreprocessType, DEVICE
from net import NetworkType, TripletNet
from dataset import DATASET
from utils.data_preprocessor import load_generate_triplet
from pipeline.prune_builder import build_pruned_embedding_net

# ── Model paths ────────────────────────────────────────────────────────
PIPELINE_DIR = _parent / "pipeline"
RUNS_ITER = PIPELINE_DIR / "runs_iterative" / "run_004_ResNet_iterative"

# 4 models to visualize
MODEL_PATHS = [
    (RUNS_ITER / "iteration_01" / "pruned" / "weights" / "Extractor_best.pth", True, "Teacher"),      # teacher
    (RUNS_ITER / "iteration_02" / "pruned" / "weights" / "Extractor_best.pth", True, "Student R1"),   # student round 1
    (RUNS_ITER / "iteration_03" / "pruned" / "weights" / "Extractor_best.pth", True, "Student R2"),   # student round 2
    (RUNS_ITER / "iteration_04" / "pruned" / "weights" / "Extractor_best.pth", True, "Student R3"),   # student round 3
]

# ── Output ─────────────────────────────────────────────────────────────
OUT_DIR = _parent / "paper" / "figures"
os.makedirs(str(OUT_DIR), exist_ok=True)
OUTPUT = OUT_DIR / "feature_visualization.pdf"

# ── Constants ──────────────────────────────────────────────────────────
PREPROCESS = PreprocessType.STFT
NET_TYPE = NetworkType.ResNet
MAX_DEVICES = 30         # max devices to show (colors)
MAX_SAMPLES_PER_DEV = 10  # samples per device
N_TSNE_ITER = 1000       # t-SNE iterations


def load_model(path, is_pruned):
    """Load a model (original or pruned format).

    Handles two pruned checkpoint formats:
      Format A (dict): {'state_dict': ..., 'channels': ..., 'embedding_dim': ...}
      Format B (OrderedDict): keys with 'embedding_net.' prefix, full-size model
    """
    saved = torch.load(str(path), map_location=DEVICE, weights_only=True)

    if is_pruned:
        # ── Format A: dict with metadata ──
        if isinstance(saved, dict) and 'state_dict' in saved:
            channels = saved.get("channels", None)
            embedding_dim = saved.get("embedding_dim", 8)
            net_type_str = saved.get("net_type", "ResNet")

            if channels is not None:
                emb_net = build_pruned_embedding_net(
                    net_type_str, PREPROCESS.in_channels, channels, embedding_dim)
                emb_net.load_state_dict(saved["state_dict"])
            else:
                raise ValueError("Pruned model missing 'channels' info")
        # ── Format B: raw OrderedDict with 'embedding_net.' prefix ──
        else:
            state_dict = {k.replace('embedding_net.', ''): v for k, v in saved.items()}
            conv1_out = state_dict['conv1.weight'].shape[0]
            l1_out    = state_dict['layer1.conv2.weight'].shape[0]
            l2_out    = state_dict['layer2.conv2.weight'].shape[0]
            l3_out    = state_dict['layer3.conv2.weight'].shape[0]
            l4_out    = state_dict['layer4.conv2.weight'].shape[0]
            channels = [conv1_out, l1_out, l2_out, l3_out, l4_out]
            embedding_dim = state_dict['fc.weight'].shape[0]
            emb_net = build_pruned_embedding_net(
                "ResNet", PREPROCESS.in_channels, channels, embedding_dim)
            emb_net.load_state_dict(state_dict)

        model = TripletNet(net_type=NET_TYPE, in_channels=PREPROCESS.in_channels)
        model.embedding_net = emb_net
    else:
        model = TripletNet(net_type=NET_TYPE, in_channels=PREPROCESS.in_channels)
        model.load_state_dict(saved)

    model = model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, data):
    """Extract embedding vectors from a batch of data."""
    model.eval()
    data = data.to(DEVICE)
    with torch.no_grad():
        embeddings = model.embedding_net(data)
    return embeddings.cpu().numpy()


def main():
    print("Loading test data...")
    file_path = str(DATASET["Test"]["seen"].path)
    label, triplet = load_generate_triplet(
        file_path, np.arange(0, 40), np.arange(0, 10),
        PREPROCESS, snr_range=None,
    )
    anchor_data = triplet[0]
    anchor_labels = label.numpy() if torch.is_tensor(label) else np.array(label)
    # Ensure 1D
    anchor_labels = np.squeeze(anchor_labels)
    print(f"Labels shape: {anchor_labels.shape}, unique: {len(np.unique(anchor_labels))}")

    # Subset: pick MAX_DEVICES devices, MAX_SAMPLES_PER_DEV each
    rng = np.random.RandomState(42)
    unique_devs = np.unique(anchor_labels)[:MAX_DEVICES]
    mask = np.isin(anchor_labels, unique_devs)
    sub_idx = np.where(mask)[0]
    # For each device, pick MAX_SAMPLES_PER_DEV
    keep = []
    for d in unique_devs:
        d_idx = np.where(anchor_labels == d)[0]
        d_idx = rng.choice(d_idx, min(MAX_SAMPLES_PER_DEV, len(d_idx)), replace=False)
        keep.extend(d_idx)
    keep = np.array(keep)
    keep.sort()

    data_subset = anchor_data[keep]
    labels_subset = anchor_labels[keep]
    print(f"Subset: {len(data_subset)} samples, {len(unique_devs)} devices × ≤{MAX_SAMPLES_PER_DEV}")

    # ── Load models & extract embeddings ───────────────────────────
    embeddings_list = []
    model_labels = []

    for path, is_pruned, label_name in MODEL_PATHS:
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue
        print(f"Loading: {label_name}  ({'pruned' if is_pruned else 'original'})")
        model = load_model(path, is_pruned)
        emb = extract_embeddings(model, data_subset)
        embeddings_list.append(emb)
        model_labels.append(label_name)
        print(f"  Embedding dim: {emb.shape[1]}")

    # ── t-SNE on each embedding space (separate fit) ───────────────
    print("\nRunning t-SNE...")
    tsne_results = []
    for i, emb in enumerate(embeddings_list):
        print(f"  t-SNE on {model_labels[i]}  ({emb.shape[1]}-dim → 2-dim)")
        tsne = TSNE(n_components=2, perplexity=min(30, emb.shape[0] // 3),
                    random_state=42, max_iter=N_TSNE_ITER, n_jobs=1)
        emb_2d = tsne.fit_transform(emb)
        tsne_results.append(emb_2d)

    # ── Plot (IEEE single-column, authored at final size) ─────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,          # embed TrueType (IEEE PDF eXpress friendly)
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(4, 2),
                             constrained_layout=True)
    axes = axes.ravel()

    # Device identity is an ordered index (0..N-1): sample one high-variation
    # colormap evenly instead of cycling 30 ad-hoc hues; a colorbar replaces
    # the per-device legend (30 entries would be unreadable at this size).
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = plt.get_cmap("turbo", len(unique_devs))
    disc_cmap = ListedColormap([cmap(i) for i in range(len(unique_devs))])
    norm = BoundaryNorm(np.arange(-0.5, len(unique_devs) + 0.5, 1.0), disc_cmap.N)
    dev_to_idx = {d: i for i, d in enumerate(unique_devs)}
    sample_idx = np.array([dev_to_idx[d] for d in np.ravel(labels_subset)])

    for i, (ax, emb_2d, lbl) in enumerate(zip(axes, tsne_results, model_labels)):
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_idx,
                        cmap=disc_cmap, norm=norm, s=2, alpha=0.9,
                        linewidths=0)
        ax.set_title(f"({chr(97 + i)}) {lbl}", fontsize=7.5, pad=2.5)
        ax.set_xticks([])          # t-SNE axes carry no meaningful scale
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        if i >= 2:
            ax.set_xlabel("t-SNE 1")
        if i % 2 == 0:
            ax.set_ylabel("t-SNE 2")

    for ax in axes[len(tsne_results):]:   # hide panels if a model was missing
        ax.set_visible(False)

    tick_pos = np.arange(0, len(unique_devs), 5)
    cbar = fig.colorbar(sc, ax=list(axes), shrink=0.85, pad=0.03,
                        ticks=tick_pos, aspect=28)
    cbar.set_label("Device index", fontsize=7)
    cbar.ax.set_yticklabels([str(unique_devs[p]) for p in tick_pos])
    cbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(str(OUTPUT))
    print(f"\nSaved: {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
