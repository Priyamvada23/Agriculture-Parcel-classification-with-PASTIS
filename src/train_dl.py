"""Training loop for the deep-learning track (Temporal-Attention U-Net).

Designed to run on GPU (e.g. Google Colab: upload/mount PASTIS_subset,
`pip install -r requirements.txt`, `python src/train.py --track dl`) but
also runs on CPU (much slower — see README for a reduced-epoch CPU smoke
test). Falls back to CPU automatically if CUDA is unavailable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_loading import PROJECT_ROOT, load_s2, load_target
from evaluate import compute_classification_metrics, evaluate_and_save, save_metrics
from model_dl import TemporalAttentionUNet, count_parameters
from model_ml import read_split
from preprocessing import compute_band_normalization_stats, to_reflectance


class PastisSegDataset(Dataset):
    """Yields (x: (T,C,H,W) float32, y: (H,W) int64) for one patch per item."""

    def __init__(self, config: dict, patch_ids: list[int], band_mean, band_std, augment: bool = False):
        self.config = config
        self.patch_ids = patch_ids
        self.mean = np.asarray(band_mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(band_std, dtype=np.float32).reshape(1, -1, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        s2 = to_reflectance(load_s2(self.config, pid))  # (T, C, H, W)
        s2 = (s2 - self.mean) / self.std
        target = load_target(self.config, pid).astype(np.int64)  # (H, W)

        if self.augment:
            if np.random.rand() < 0.5:
                s2, target = s2[:, :, :, ::-1], target[:, ::-1]
            if np.random.rand() < 0.5:
                s2, target = s2[:, :, ::-1, :], target[::-1, :]
            k = np.random.randint(4)
            if k:
                s2 = np.rot90(s2, k, axes=(2, 3))
                target = np.rot90(target, k, axes=(0, 1))

        x = torch.from_numpy(np.ascontiguousarray(s2)).float()
        y = torch.from_numpy(np.ascontiguousarray(target)).long()
        return x, y, pid


def get_device(config: dict) -> torch.device:
    wanted = config["model_dl"]["device"]
    if wanted == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available -> falling back to CPU (this will be slow; "
              "see README for running the full training on Google Colab).")
        return torch.device("cpu")
    return torch.device(wanted)


def compute_class_weights(config: dict, train_ids: list[int]) -> torch.Tensor:
    from data_loading import class_pixel_counts

    counts = class_pixel_counts(config, train_ids)
    n_classes = len(config["classes"])
    void = config["void_class"]
    freqs = np.array([counts.get(c, 0) for c in range(n_classes)], dtype=np.float64)
    freqs[void] = 0
    total = freqs.sum()
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = freqs > 0
    weights[nonzero] = total / (nonzero.sum() * freqs[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_dataset(model, dataset, device, void_class, n_classes, collect_predictions=False):
    model.eval()
    y_true_all, y_pred_all = [], []
    pred_maps = {}
    for x, y, pid in DataLoader(dataset, batch_size=1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)  # (1, H, W)

        y_np, pred_np = y.cpu().numpy().reshape(-1), pred.cpu().numpy().reshape(-1)
        mask = y_np != void_class
        y_true_all.append(y_np[mask])
        y_pred_all.append(pred_np[mask])
        if collect_predictions:
            pred_maps[int(pid[0])] = pred.cpu().numpy()[0]
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), pred_maps


def run_dl_track(config: dict, max_epochs: int | None = None, max_train_patches: int | None = None) -> None:
    seed = config["split"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(config)
    print(f"=== Deep learning track: Temporal-Attention U-Net (device={device}) ===")

    train_ids = read_split(config, "train")
    val_ids = read_split(config, "val")
    test_ids = read_split(config, "test")
    if max_train_patches:
        train_ids = train_ids[:max_train_patches]

    print("Computing per-band normalization stats from the training split...")
    stats = compute_band_normalization_stats(config, train_ids)
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(models_dir / "dl_band_stats.npz", mean=stats["mean"], std=stats["std"])

    train_ds = PastisSegDataset(config, train_ids, stats["mean"], stats["std"], augment=True)
    val_ds = PastisSegDataset(config, val_ids, stats["mean"], stats["std"], augment=False)
    test_ds = PastisSegDataset(config, test_ids, stats["mean"], stats["std"], augment=False)

    dl_cfg = config["model_dl"]
    train_loader = DataLoader(train_ds, batch_size=dl_cfg["batch_size"], shuffle=True, num_workers=0)

    n_classes = len(config["classes"])
    void_class = config["void_class"]
    model = TemporalAttentionUNet(
        in_channels=config["data"]["n_bands"],
        n_classes=n_classes,
        temporal_hidden_dim=dl_cfg["temporal_hidden_dim"],
        unet_base_channels=dl_cfg["unet_base_channels"],
    ).to(device)
    print(f"model parameters: {count_parameters(model):,}")

    class_weights = compute_class_weights(config, train_ids).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=void_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg["learning_rate"], weight_decay=dl_cfg["weight_decay"])

    epochs = max_epochs if max_epochs is not None else dl_cfg["epochs"]
    best_val_macro_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)

        y_true, y_pred, _ = evaluate_dataset(model, val_ds, device, void_class, n_classes)
        # Checkpoint selection uses the same (crop-only) protocol as the
        # reported metrics, so the selected model is the one that is best by
        # the criterion actually being reported.
        val_metrics = compute_classification_metrics(
            y_true, y_pred, config["classes"],
            ignore_classes=config["evaluation"]["ignore_classes"])
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"]})

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "config": dl_cfg}, models_dir / "dl_model.pt")
    print(f"saved best checkpoint (val_macro_f1={best_val_macro_f1:.4f}) to {models_dir/'dl_model.pt'}")

    import pandas as pd
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    pd.DataFrame(history).to_csv(metrics_dir / "dl_training_history.csv", index=False)

    predictions_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ds in (("val", val_ds), ("test", test_ds)):
        y_true, y_pred, pred_maps = evaluate_dataset(
            model, ds, device, void_class, n_classes, collect_predictions=(split_name == "test")
        )
        evaluate_and_save(config, y_true, y_pred, pred_maps,
                          tag="dl", split=split_name)


if __name__ == "__main__":
    from data_loading import load_config

    cfg = load_config()
    run_dl_track(cfg)
