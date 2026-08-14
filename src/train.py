"""Train a crop-classification model (assignment task D).

Usage:
    python train.py --track ml     # Random Forest + XGBoost pixel classifiers (this machine, CPU)
    python train.py --track dl     # U-Net-style temporal segmentation model (see model_dl.py;
                                    # intended for a GPU runtime such as Google Colab)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from data_loading import PROJECT_ROOT, load_config
from evaluate import compute_classification_metrics, evaluate_and_save, save_metrics
import model_ml as mlm


def run_ml_track(config: dict) -> None:
    print("=== Traditional ML track: Random Forest + XGBoost pixel classifiers ===")
    train_ids = mlm.read_split(config, "train")
    val_ids = mlm.read_split(config, "val")
    test_ids = mlm.read_split(config, "test")
    void_class = config["void_class"]
    class_names = config["classes"]

    print(f"Building class-stratified training feature table from {len(train_ids)} train patches...")
    t0 = time.time()
    train_df = mlm.build_training_table(config, train_ids)
    feat_cols = mlm.feature_columns(train_df)
    X_train, y_train = train_df[feat_cols].values, train_df["label"].values
    print(f"  {X_train.shape[0]} rows x {X_train.shape[1]} features "
          f"({len(np.unique(y_train))} classes) in {time.time()-t0:.1f}s")

    results = {}

    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = mlm.train_random_forest(X_train, y_train, config)
    print(f"  done in {time.time()-t0:.1f}s")
    mlm.save_model(rf, PROJECT_ROOT / config["output"]["models_dir"] / "rf_model.joblib",
                    extra={"feature_names": feat_cols})
    results["rf"] = ("rf", lambda X: rf.predict(X))

    print("\nTraining XGBoost...")
    t0 = time.time()
    xgb_clf, idx_to_class = mlm.train_xgboost(X_train, y_train, config)
    print(f"  done in {time.time()-t0:.1f}s")
    mlm.save_model(xgb_clf, PROJECT_ROOT / config["output"]["models_dir"] / "xgb_model.joblib",
                    extra={"feature_names": feat_cols, "idx_to_class": idx_to_class})
    results["xgb"] = ("xgb", lambda X: mlm.predict_xgboost(xgb_clf, X, idx_to_class))

    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    predictions_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for name, (tag, predict_fn) in results.items():
        for split_name, ids in (("val", val_ids), ("test", test_ids)):
            print(f"\nEvaluating {name} on {split_name} ({len(ids)} patches, full pixel grid)...")
            y_true, y_pred, pred_maps = mlm.evaluate_full_patches(
                config, predict_fn, feat_cols, ids, void_class
            )
            metrics = evaluate_and_save(config, y_true, y_pred, pred_maps,
                                        tag=f"ml_{tag}", split=split_name)
            if split_name == "test":
                comparison_rows.append(
                    {"model": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                     "weighted_f1": metrics["weighted_f1"], "mean_iou": metrics["mean_iou"]}
                )

    import pandas as pd
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(metrics_dir / "ml_model_comparison_test.csv", index=False)
    print("\n=== ML track comparison (test set) ===")
    print(comp_df.to_string(index=False))


def run_ndvi_track(config: dict) -> None:
    """Track C: Random Forest on the ordered NDVI time series.

    Same split, same sampling, same RF hyperparameters and same metrics as the
    Track A Random Forest -- the only thing that changes is the per-pixel
    feature representation (46 ordered NDVI values instead of 60 temporal
    statistics). That makes the two directly comparable as a test of whether
    preserving phenological *timing* helps.
    """
    from preprocessing import patch_ndvi_series

    print("=== NDVI track: Random Forest on the ordered NDVI time series ===")
    train_ids = mlm.read_split(config, "train")
    val_ids = mlm.read_split(config, "val")
    test_ids = mlm.read_split(config, "test")
    void_class = config["void_class"]
    class_names = config["classes"]

    print(f"Building NDVI time-series table from {len(train_ids)} train patches...")
    t0 = time.time()
    train_df = mlm.build_training_table(config, train_ids, feature_fn=patch_ndvi_series)
    feat_cols = mlm.feature_columns(train_df)
    X_train, y_train = train_df[feat_cols].values, train_df["label"].values
    print(f"  {X_train.shape[0]} rows x {X_train.shape[1]} features "
          f"({len(np.unique(y_train))} classes) in {time.time()-t0:.1f}s")

    from sklearn.ensemble import RandomForestClassifier
    print("\nTraining Random Forest on NDVI series...")
    t0 = time.time()
    rf = RandomForestClassifier(**config["model_ndvi"]["random_forest"])
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s")

    mlm.save_model(rf, PROJECT_ROOT / config["output"]["models_dir"] / "ndvi_rf_model.joblib",
                   extra={"feature_names": feat_cols})

    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    predictions_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in (("val", val_ids), ("test", test_ids)):
        print(f"\nEvaluating on {split_name} ({len(ids)} patches, full pixel grid)...")
        y_true, y_pred, pred_maps = mlm.evaluate_full_patches(
            config, lambda X: rf.predict(X), feat_cols, ids, void_class,
            feature_fn=patch_ndvi_series,
        )
        evaluate_and_save(config, y_true, y_pred, pred_maps,
                          tag="ndvi_rf", split=split_name)

    # Feature importance by acquisition date -- shows which points in the
    # season the model actually keys on (a phenology read-out for free).
    import pandas as pd
    from data_loading import load_metadata, get_dates
    dates = get_dates(load_metadata(config), train_ids[0])
    imp = pd.DataFrame({
        "feature": feat_cols,
        "date": [dates[int(f.split("t")[-1])] for f in feat_cols],
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(metrics_dir / "ndvi_rf_feature_importance.csv", index=False)
    print("\nTop 10 most informative acquisition dates:")
    print(imp.head(10).to_string(index=False))


def run_ndvi_dl_track(config: dict) -> None:
    """Track D: 1D temporal CNN on the ordered NDVI series (neural twin of Track C).

    Uses the same feature function, the same split and the same stratified
    pixel sampling as Track C, and is evaluated through the same
    `evaluate_full_patches` path as every other track, so the only variable
    versus Track C is the model class.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from model_ndvi_dl import TempCNN, count_parameters
    from preprocessing import patch_ndvi_series
    from train_dl import compute_class_weights, get_device

    cfg_d = config["model_ndvi_dl"]
    device = get_device({"model_dl": {"device": cfg_d["device"]}})
    seed = config["split"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"=== Track D: 1D temporal CNN on the NDVI series (device={device}) ===")
    train_ids = mlm.read_split(config, "train")
    val_ids = mlm.read_split(config, "val")
    test_ids = mlm.read_split(config, "test")
    void_class = config["void_class"]
    n_classes = len(config["classes"])

    print(f"Building NDVI tables (train={len(train_ids)}, val={len(val_ids)} patches)...")
    t0 = time.time()
    train_df = mlm.build_training_table(config, train_ids, feature_fn=patch_ndvi_series)
    val_df = mlm.build_training_table(config, val_ids, feature_fn=patch_ndvi_series)
    feat_cols = mlm.feature_columns(train_df)
    Xtr = torch.tensor(train_df[feat_cols].values, dtype=torch.float32)
    ytr = torch.tensor(train_df["label"].values, dtype=torch.long)
    Xva = torch.tensor(val_df[feat_cols].values, dtype=torch.float32)
    yva = torch.tensor(val_df["label"].values, dtype=torch.long)
    print(f"  train {tuple(Xtr.shape)}, val {tuple(Xva.shape)} in {time.time()-t0:.1f}s")

    model = TempCNN(
        n_timesteps=len(feat_cols), n_classes=n_classes,
        channels=cfg_d["channels"], hidden_dim=cfg_d["hidden_dim"], dropout=cfg_d["dropout"],
    ).to(device)
    print(f"model parameters: {count_parameters(model):,}")

    # Same inverse-frequency weighting the U-Net uses, so class-imbalance
    # handling matches across the neural tracks (and mirrors Track C's
    # class_weight='balanced').
    weights = compute_class_weights(config, train_ids).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=void_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_d["learning_rate"],
                                 weight_decay=cfg_d["weight_decay"])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg_d["batch_size"], shuffle=True)

    best_macro_f1, best_state, history = -1.0, None, []
    for epoch in range(1, cfg_d["epochs"] + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xtr)

        # Epoch monitoring uses the *sampled* val table (fast); the reported
        # metrics below are always recomputed on the full val/test pixel grid.
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xva), 8192):
                preds.append(model(Xva[i:i+8192].to(device))[:, :void_class].argmax(1).cpu())
            preds = torch.cat(preds).numpy()
        # Same (crop-only) protocol as the reported metrics, so checkpoint
        # selection optimises the criterion actually being reported.
        m = compute_classification_metrics(yva.numpy(), preds, config["classes"],
                                           ignore_classes=config["evaluation"]["ignore_classes"])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_sampled_accuracy": m["accuracy"], "val_sampled_macro_f1": m["macro_f1"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{cfg_d['epochs']}  loss={train_loss:.4f}  "
                  f"val(sampled) acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1, best_state = m["macro_f1"], {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg_d,
                "feature_names": feat_cols}, models_dir / "ndvi_dl_model.pt")
    print(f"saved best checkpoint (val-sampled macro-F1={best_macro_f1:.4f})")

    import pandas as pd
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    pd.DataFrame(history).to_csv(metrics_dir / "ndvi_dl_training_history.csv", index=False)

    # Wrap the torch model as a numpy predict_fn so it flows through exactly
    # the same full-patch evaluation path as the tree-based tracks.
    def predict_fn(X: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i+8192], dtype=torch.float32, device=device)
                out.append(model(xb)[:, :void_class].argmax(1).cpu().numpy())
        return np.concatenate(out)

    predictions_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in (("val", val_ids), ("test", test_ids)):
        print(f"\nEvaluating on {split_name} ({len(ids)} patches, full pixel grid)...")
        y_true, y_pred, pred_maps = mlm.evaluate_full_patches(
            config, predict_fn, feat_cols, ids, void_class, feature_fn=patch_ndvi_series
        )
        evaluate_and_save(config, y_true, y_pred, pred_maps,
                          tag="ndvi_dl", split=split_name)


def run_dl_track(config: dict) -> None:
    from train_dl import run_dl_track as _run
    _run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["ml", "ndvi", "ndvi_dl", "dl"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.track == "ml":
        run_ml_track(cfg)
    elif args.track == "ndvi":
        run_ndvi_track(cfg)
    elif args.track == "ndvi_dl":
        run_ndvi_dl_track(cfg)
    else:
        run_dl_track(cfg)
