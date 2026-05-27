# main.py
# Usage:
#   python main.py
#
# Required files:
#   train.csv
#   test.csv
#   preprocess.py
#
# This script automatically runs preprocess.py if processed files are missing.
# It does NOT need sample_submission.csv.
#
# Output:
#   submission.csv

import os
import random
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42
N_SPLITS = 5
EPOCHS = 20
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 4

TARGET = "PitNextLap"
ID_COL = "id"

TRAIN_PATH = "processed_train.csv"
TEST_PATH = "processed_test.csv"
RAW_TEST_PATH = "test.csv"
OUT_PATH = "submission.csv"


def ensure_processed_data():
    need_preprocess = (
        not os.path.exists(TRAIN_PATH)
        or not os.path.exists(TEST_PATH)
    )

    if need_preprocess:
        print("Processed files not found. Running preprocess.py ...")
        result = subprocess.run([sys.executable, "preprocess.py"])
        if result.returncode != 0:
            raise RuntimeError("preprocess.py failed. Please check preprocessing error messages.")
    else:
        print("Processed files found. Skip preprocessing.")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TabularDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class TabularNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_loader(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            preds.append(prob.cpu().numpy())

    return np.concatenate(preds)


def train_one_fold(fold, x_train, y_train, x_val, y_val, x_test, pos_weight, device):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TabularDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(TabularDataset(x_test_scaled), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TabularNN(x_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_auc = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        val_pred = predict_loader(model, val_loader, device)
        val_auc = roc_auc_score(y_val, val_pred)
        scheduler.step(val_auc)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Fold {fold} | Epoch {epoch:02d} | loss={avg_loss:.5f} | val_auc={val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"Fold {fold} early stop. Best AUC = {best_auc:.6f}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    val_pred = predict_loader(model, val_loader, device)
    test_pred = predict_loader(model, test_loader, device)

    return best_auc, val_pred, test_pred


def main():
    ensure_processed_data()
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train[TARGET].values.astype(np.float32)
    test_ids = test[ID_COL].copy()

    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"processed_test.csv missing columns: {missing[:10]}")

    # Make sure Driver is not accidentally used
    driver_like = [c for c in feature_cols if "driver" in c.lower()]
    if driver_like:
        raise ValueError(f"Driver-related columns should not be used, but found: {driver_like[:10]}")

    x = train[feature_cols].values.astype(np.float32)
    x_test = test[feature_cols].values.astype(np.float32)

    print("Processed train shape:", train.shape)
    print("Processed test shape :", test.shape)
    print("Feature dim:", x.shape[1])
    print("Positive ratio:", y.mean())

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    pos_weight = neg / max(pos, 1)
    print("pos_weight:", pos_weight)

    oof = np.zeros(len(x), dtype=np.float32)
    test_preds = np.zeros(len(x_test), dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        print("=" * 60)
        print(f"Fold {fold}/{N_SPLITS}")

        best_auc, val_pred, test_pred = train_one_fold(
            fold=fold,
            x_train=x[tr_idx],
            y_train=y[tr_idx],
            x_val=x[va_idx],
            y_val=y[va_idx],
            x_test=x_test,
            pos_weight=pos_weight,
            device=device
        )

        oof[va_idx] = val_pred
        test_preds += test_pred / N_SPLITS
        aucs.append(best_auc)

    overall_auc = roc_auc_score(y, oof)

    print("=" * 60)
    print("Fold AUCs:", [round(a, 6) for a in aucs])
    print("OOF AUC:", round(overall_auc, 6))

    # Kaggle submission format: id, PitNextLap
    submission = pd.DataFrame({
        ID_COL: test_ids.values,
        TARGET: np.clip(test_preds, 0, 1)
    })

    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
