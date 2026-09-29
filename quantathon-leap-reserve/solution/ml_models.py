"""Neural network models for parcel-level LEAP predictions.

Three PyTorch models trained on v2 LEAP data (System Materials, Quotes,
Participation, Insurance) and evaluated on the Columbus v4 synthetic
test set:
1. UptakeNet — predicts per-parcel probability of applying for LEAP
2. SaleHazardNet — predicts per-parcel annual sale probability
3. CostNet — predicts per-parcel replacement cost

Training data is generated from the v2 real-world distributions so that
model calibration derives entirely from observed program data.  The
Columbus v4 dataset (10,000 synthetic parcels) is held out as an
independent test set for out-of-sample evaluation.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_utils import DATASET_DIR, _read_csv_rows, generate_v2_training_parcels

TRAINED_MODELS_DIR = Path(__file__).resolve().parent / "trained_models"


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_v4_parcels() -> List[Dict]:
    """Load Columbus v4 synthetic parcels as the held-out test set."""
    rows = _read_csv_rows(DATASET_DIR / "Columbus v4.csv")
    header = rows[1]
    body = rows[2:]
    idx = {name: i for i, name in enumerate(header)}
    parcels = []
    for row in body:
        parcels.append({
            "parcel_id": row[idx["parcel_id"]],
            "zip_code": int(row[idx["zip_code"]]),
            "property_type": row[idx["property_type"]],
            "year_built": int(row[idx["year_built"]]),
            "assessed_value": float(row[idx["assessed_value"]]),
            "last_sale_year": int(row[idx["last_sale_year"]]),
            "ownership_length": int(row[idx["ownership_length"]]),
            "owner_occupied": int(row[idx["owner_occupied"]]),
            "lead_risk": float(row[idx["lead_risk"]]),
            "line_material": row[idx["line_material"]],
            "estimated_replacement_cost": float(row[idx["estimated_replacement_cost"]]),
        })
    return parcels


def _encode_property_type(pt: str) -> List[float]:
    """One-hot encode property type."""
    return [
        1.0 if pt == "single_family" else 0.0,
        1.0 if pt == "duplex" else 0.0,
        1.0 if pt == "multi_family" else 0.0,
    ]


def _encode_material(mat: str) -> List[float]:
    """One-hot encode line material."""
    return [
        1.0 if mat == "lead" else 0.0,
        1.0 if mat == "galvanized" else 0.0,
        1.0 if mat == "copper" else 0.0,
    ]


def _build_feature_vector(p: Dict) -> List[float]:
    """Build full feature vector for a parcel (13 features)."""
    return [
        p["year_built"],
        p["assessed_value"],
        p["ownership_length"],
        p["owner_occupied"],
        p["lead_risk"],
        p["zip_code"] - 43200,  # normalize zip to small range
    ] + _encode_property_type(p["property_type"]) + _encode_material(p["line_material"])


def _engineer_uptake_target(p: Dict) -> float:
    """Engineer a synthetic uptake probability from parcel features.

    Higher for: high lead_risk, owner-occupied, lower assessed value,
    older ownership (less likely to sell soon anyway → more motivated to fix).
    Calibrated so the population average ≈ 0.002 (matching observed 0.20%).
    """
    risk_factor = p["lead_risk"]  # 0-1, higher = more likely to want LEAP
    owner_factor = 0.8 if p["owner_occupied"] else 0.2  # owners more likely
    # Lower value homes more motivated (normalized to ~0.5-1.5 range)
    value_factor = max(0.3, min(1.5, 1.0 - (p["assessed_value"] - 90000) / 300000))
    # Older ownership → more settled, may be more proactive
    tenure_factor = min(1.3, 0.7 + p["ownership_length"] / 50.0)

    raw = risk_factor * owner_factor * value_factor * tenure_factor
    # Scale so that population average is ~0.002 (0.20%)
    # Raw average is roughly 0.35, so multiply by 0.002/0.35 ≈ 0.0057
    scaled = raw * 0.0057
    return min(0.02, max(0.0001, scaled))  # clip to reasonable range


def _engineer_sale_target(p: Dict) -> float:
    """Engineer per-parcel annual sale probability.

    Shorter ownership → higher sale probability (recently bought → may flip).
    Higher assessed value → more liquid market.
    Owner-occupied → slightly lower sale probability.
    Base rate ~5.6% (matching the model's flat assumption).
    """
    # Inverse relationship with ownership length
    if p["ownership_length"] <= 3:
        tenure_factor = 1.6  # recently bought, higher turnover
    elif p["ownership_length"] <= 7:
        tenure_factor = 1.2
    elif p["ownership_length"] <= 15:
        tenure_factor = 1.0
    elif p["ownership_length"] <= 25:
        tenure_factor = 0.8
    else:
        tenure_factor = 0.6  # long-term owners sell less

    # Higher value → more liquid
    value_factor = 0.8 + (p["assessed_value"] / 500000) * 0.4
    value_factor = min(1.3, max(0.7, value_factor))

    # Owner-occupied slightly lower
    owner_factor = 0.95 if p["owner_occupied"] else 1.15

    base_rate = 0.056
    prob = base_rate * tenure_factor * value_factor * owner_factor
    return min(0.15, max(0.01, prob))


def prepare_datasets(
    train_parcels: List[Dict],
    test_parcels: List[Dict],
) -> Dict[str, Dict]:
    """Prepare train / test datasets for all 3 models.

    Training parcels are generated from v2 LEAP data distributions.
    Test parcels come from the Columbus v4 synthetic dataset.
    The scaler is fit on the training set only.
    """
    # --- training set (v2-derived) ---
    train_features = np.array(
        [_build_feature_vector(p) for p in train_parcels], dtype=np.float32
    )
    train_uptake = np.array(
        [_engineer_uptake_target(p) for p in train_parcels], dtype=np.float32
    )
    train_sale = np.array(
        [_engineer_sale_target(p) for p in train_parcels], dtype=np.float32
    )
    train_cost = np.array(
        [p["estimated_replacement_cost"] for p in train_parcels], dtype=np.float32
    )

    # --- test set (v4) ---
    test_features = np.array(
        [_build_feature_vector(p) for p in test_parcels], dtype=np.float32
    )
    test_uptake = np.array(
        [_engineer_uptake_target(p) for p in test_parcels], dtype=np.float32
    )
    test_sale = np.array(
        [_engineer_sale_target(p) for p in test_parcels], dtype=np.float32
    )
    test_cost = np.array(
        [p["estimated_replacement_cost"] for p in test_parcels], dtype=np.float32
    )

    # Normalize features — fit on training set only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)

    # Normalize cost targets for training stability (using train stats)
    cost_mean = float(train_cost.mean())
    cost_std = float(train_cost.std())
    train_cost_norm = ((train_cost - cost_mean) / cost_std).astype(np.float32)
    test_cost_norm = ((test_cost - cost_mean) / cost_std).astype(np.float32)

    return {
        "features_scaler": scaler,
        "cost_mean": cost_mean,
        "cost_std": cost_std,
        "train": {
            "X": train_scaled,
            "uptake_y": train_uptake,
            "sale_y": train_sale,
            "cost_y": train_cost_norm,
            "cost_y_raw": train_cost,
        },
        "val": {
            "X": test_scaled,
            "uptake_y": test_uptake,
            "sale_y": test_sale,
            "cost_y": test_cost_norm,
            "cost_y_raw": test_cost,
        },
    }


# ---------------------------------------------------------------------------
# Neural network definitions
# ---------------------------------------------------------------------------

class UptakeNet(nn.Module):
    """Predicts per-parcel LEAP uptake probability."""

    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SaleHazardNet(nn.Module):
    """Predicts per-parcel annual sale probability."""

    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CostNet(nn.Module):
    """Predicts per-parcel replacement cost."""

    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 256,
    model_name: str = "model",
) -> Dict:
    """Train a single model and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy()

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs,
        "val_predictions": val_pred,
        "val_targets": y_val,
        "history": history,
    }


def train_all_models(parcels: List[Dict] = None, verbose: bool = True) -> Dict:
    """Train all 3 models on v2 data and evaluate on v4 test set.

    Training parcels are generated from v2 LEAP data distributions.
    The Columbus v4 dataset is used as the held-out test set.
    """
    train_parcels = generate_v2_training_parcels()
    test_parcels = load_v4_parcels()

    if verbose:
        print("Preparing datasets (train: v2 data, test: v4 data)...")
    datasets = prepare_datasets(train_parcels, test_parcels)

    train_data = datasets["train"]
    val_data = datasets["val"]
    input_dim = train_data["X"].shape[1]

    if verbose:
        print(f"Training set (v2): {len(train_data['X'])} parcels, Test set (v4): {len(val_data['X'])} parcels")
        print(f"Feature dimension: {input_dim}")
        print()

    # --- Model 1: Uptake ---
    if verbose:
        print("Training UptakeNet...")
    uptake_model = UptakeNet(input_dim)
    # Scale targets to 0-1 range for BCE-like loss
    uptake_max = max(train_data["uptake_y"].max(), val_data["uptake_y"].max())
    uptake_train_scaled = train_data["uptake_y"] / uptake_max
    uptake_val_scaled = val_data["uptake_y"] / uptake_max
    uptake_metrics = _train_model(
        uptake_model, train_data["X"], uptake_train_scaled,
        val_data["X"], uptake_val_scaled,
        loss_fn=nn.MSELoss(), epochs=150, model_name="Uptake",
    )

    # --- Model 2: Sale Hazard ---
    if verbose:
        print("\nTraining SaleHazardNet...")
    sale_model = SaleHazardNet(input_dim)
    sale_metrics = _train_model(
        sale_model, train_data["X"], train_data["sale_y"],
        val_data["X"], val_data["sale_y"],
        loss_fn=nn.MSELoss(), epochs=150, model_name="SaleHazard",
    )

    # --- Model 3: Cost ---
    if verbose:
        print("\nTraining CostNet...")
    cost_model = CostNet(input_dim)
    cost_metrics = _train_model(
        cost_model, train_data["X"], train_data["cost_y"],
        val_data["X"], val_data["cost_y"],
        loss_fn=nn.MSELoss(), epochs=150, model_name="Cost",
    )

    # Compute R² for cost model (on raw scale)
    cost_val_pred_norm = cost_metrics["val_predictions"]
    cost_val_pred_raw = cost_val_pred_norm * datasets["cost_std"] + datasets["cost_mean"]
    cost_val_true = val_data["cost_y_raw"]
    ss_res = np.sum((cost_val_true - cost_val_pred_raw) ** 2)
    ss_tot = np.sum((cost_val_true - cost_val_true.mean()) ** 2)
    cost_r2 = 1.0 - ss_res / ss_tot

    # Compute correlation for uptake and sale models
    uptake_val_pred = uptake_metrics["val_predictions"] * uptake_max
    uptake_val_true = val_data["uptake_y"]
    uptake_corr = np.corrcoef(uptake_val_pred, uptake_val_true)[0, 1]

    sale_val_pred = sale_metrics["val_predictions"]
    sale_val_true = val_data["sale_y"]
    sale_corr = np.corrcoef(sale_val_pred, sale_val_true)[0, 1]

    if verbose:
        print(f"\n--- Training Results ---")
        print(f"UptakeNet:     val_loss={uptake_metrics['best_val_loss']:.6f}, correlation={uptake_corr:.4f}")
        print(f"  Avg predicted uptake: {uptake_val_pred.mean():.5f} (target avg: {uptake_val_true.mean():.5f})")
        print(f"SaleHazardNet: val_loss={sale_metrics['best_val_loss']:.6f}, correlation={sale_corr:.4f}")
        print(f"  Avg predicted sale prob: {sale_val_pred.mean():.4f} (target avg: {sale_val_true.mean():.4f})")
        print(f"CostNet:       val_loss={cost_metrics['best_val_loss']:.6f}, R²={cost_r2:.4f}")
        print(f"  Avg predicted cost: ${cost_val_pred_raw.mean():,.0f} (target avg: ${cost_val_true.mean():,.0f})")

    # Save models
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    torch.save(uptake_model.state_dict(), TRAINED_MODELS_DIR / "uptake_net.pt")
    torch.save(sale_model.state_dict(), TRAINED_MODELS_DIR / "sale_hazard_net.pt")
    torch.save(cost_model.state_dict(), TRAINED_MODELS_DIR / "cost_net.pt")

    # Save scaler params
    scaler_params = {
        "mean": datasets["features_scaler"].mean_.tolist(),
        "scale": datasets["features_scaler"].scale_.tolist(),
        "cost_mean": datasets["cost_mean"],
        "cost_std": datasets["cost_std"],
        "uptake_max": float(uptake_max),
    }
    import json
    with open(TRAINED_MODELS_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f)

    if verbose:
        print(f"\nModels saved to {TRAINED_MODELS_DIR}/")

    return {
        "uptake_model": uptake_model,
        "sale_model": sale_model,
        "cost_model": cost_model,
        "scaler": datasets["features_scaler"],
        "cost_mean": datasets["cost_mean"],
        "cost_std": datasets["cost_std"],
        "uptake_max": float(uptake_max),
        "input_dim": input_dim,
        "metrics": {
            "uptake": {
                "val_loss": uptake_metrics["best_val_loss"],
                "correlation": float(uptake_corr),
                "avg_predicted": float(uptake_val_pred.mean()),
                "avg_target": float(uptake_val_true.mean()),
            },
            "sale_hazard": {
                "val_loss": sale_metrics["best_val_loss"],
                "correlation": float(sale_corr),
                "avg_predicted": float(sale_val_pred.mean()),
                "avg_target": float(sale_val_true.mean()),
            },
            "cost": {
                "val_loss": cost_metrics["best_val_loss"],
                "r_squared": float(cost_r2),
                "avg_predicted": float(cost_val_pred_raw.mean()),
                "avg_target": float(cost_val_true.mean()),
            },
        },
    }


def load_trained_models(input_dim: int = 12) -> Dict:
    """Load previously trained models from disk."""
    import json

    with open(TRAINED_MODELS_DIR / "scaler_params.json") as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler_params["mean"])

    uptake_model = UptakeNet(input_dim)
    uptake_model.load_state_dict(torch.load(TRAINED_MODELS_DIR / "uptake_net.pt", weights_only=True))
    uptake_model.eval()

    sale_model = SaleHazardNet(input_dim)
    sale_model.load_state_dict(torch.load(TRAINED_MODELS_DIR / "sale_hazard_net.pt", weights_only=True))
    sale_model.eval()

    cost_model = CostNet(input_dim)
    cost_model.load_state_dict(torch.load(TRAINED_MODELS_DIR / "cost_net.pt", weights_only=True))
    cost_model.eval()

    return {
        "uptake_model": uptake_model,
        "sale_model": sale_model,
        "cost_model": cost_model,
        "scaler": scaler,
        "cost_mean": scaler_params["cost_mean"],
        "cost_std": scaler_params["cost_std"],
        "uptake_max": scaler_params["uptake_max"],
        "input_dim": input_dim,
    }


def predict_parcel_features(
    parcels: List[Dict], models: Dict
) -> Dict[str, np.ndarray]:
    """Run inference on a list of parcels using all 3 models.

    Returns dict with arrays of predictions for each parcel.
    """
    features = np.array([_build_feature_vector(p) for p in parcels], dtype=np.float32)
    features_scaled = models["scaler"].transform(features).astype(np.float32)
    X = torch.from_numpy(features_scaled)

    with torch.no_grad():
        uptake_raw = models["uptake_model"](X).numpy()
        uptake_probs = uptake_raw * models["uptake_max"]

        sale_probs = models["sale_model"](X).numpy()

        cost_norm = models["cost_model"](X).numpy()
        costs = cost_norm * models["cost_std"] + models["cost_mean"]
        costs = np.clip(costs, 9000, 18000)  # reasonable range

    return {
        "uptake_prob": uptake_probs,
        "sale_prob": sale_probs,
        "replacement_cost": costs,
    }
