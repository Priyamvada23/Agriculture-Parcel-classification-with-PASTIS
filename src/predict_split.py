"""Run a saved model over a split and persist its predictions + metrics.

Inference only -- loads trained weights from `outputs/models/` rather than
retraining. Used to regenerate predictions for a split that an earlier run
did not save (older code stored test predictions only), so that every model
can be scored under the same protocol without repeating hours of training.

Usage:
    python predict_split.py --split val                  # all models
    python predict_split.py --split val --models dl      # one model
"""
from __future__ import annotations

import argparse

import numpy as np

from data_loading import PROJECT_ROOT, load_config
from evaluate import evaluate_and_save
import model_ml as mlm
from preprocessing import patch_ndvi_series, patch_pixel_features


def _tree_predictions(config, model_path, feature_fn, patch_ids, xgb=False):
    model, extra = mlm.load_model(PROJECT_ROOT / config["output"]["models_dir"] / model_path)
    feat_names = extra["feature_names"]
    if xgb:
        idx_to_class = extra["idx_to_class"]
        predict_fn = lambda X: mlm.predict_xgboost(model, X, idx_to_class)
    else:
        predict_fn = model.predict
    return mlm.evaluate_full_patches(
        config, predict_fn, feat_names, patch_ids, config["void_class"], feature_fn=feature_fn
    )


def _tempcnn_predictions(config, patch_ids):
    import torch
    from model_ndvi_dl import TempCNN

    ckpt = torch.load(PROJECT_ROOT / config["output"]["models_dir"] / "ndvi_dl_model.pt",
                      map_location="cpu", weights_only=False)
    cfg_d = ckpt["config"]
    feat_names = ckpt["feature_names"]
    model = TempCNN(n_timesteps=len(feat_names), n_classes=len(config["classes"]),
                    channels=cfg_d["channels"], hidden_dim=cfg_d["hidden_dim"],
                    dropout=cfg_d["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    void = config["void_class"]

    def predict_fn(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i + 8192], dtype=torch.float32)
                out.append(model(xb)[:, :void].argmax(1).numpy())
        return np.concatenate(out)

    return mlm.evaluate_full_patches(config, predict_fn, feat_names, patch_ids, void,
                                     feature_fn=patch_ndvi_series)


def _unet_predictions(config, patch_ids):
    import torch
    from model_dl import TemporalAttentionUNet
    from train_dl import PastisSegDataset, evaluate_dataset

    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    ckpt = torch.load(models_dir / "dl_model.pt", map_location="cpu", weights_only=False)
    cfg_d = ckpt["config"]
    model = TemporalAttentionUNet(
        in_channels=config["data"]["n_bands"], n_classes=len(config["classes"]),
        temporal_hidden_dim=cfg_d["temporal_hidden_dim"],
        unet_base_channels=cfg_d["unet_base_channels"])
    model.load_state_dict(ckpt["model_state_dict"])

    stats = np.load(models_dir / "dl_band_stats.npz")
    ds = PastisSegDataset(config, patch_ids, stats["mean"], stats["std"], augment=False)
    return evaluate_dataset(model, ds, torch.device("cpu"), config["void_class"],
                            len(config["classes"]), collect_predictions=True)


MODELS = {
    "ml_rf":   lambda c, ids: _tree_predictions(c, "rf_model.joblib", patch_pixel_features, ids),
    "ml_xgb":  lambda c, ids: _tree_predictions(c, "xgb_model.joblib", patch_pixel_features, ids, xgb=True),
    "ndvi_rf": lambda c, ids: _tree_predictions(c, "ndvi_rf_model.joblib", patch_ndvi_series, ids),
    "ndvi_dl": _tempcnn_predictions,
    "dl":      _unet_predictions,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patch_ids = mlm.read_split(cfg, args.split)

    for tag in args.models:
        print(f"\n=== {tag} on {args.split} ({len(patch_ids)} patches) ===")
        y_true, y_pred, pred_maps = MODELS[tag](cfg, patch_ids)
        evaluate_and_save(cfg, y_true, y_pred, pred_maps, tag=tag, split=args.split)
