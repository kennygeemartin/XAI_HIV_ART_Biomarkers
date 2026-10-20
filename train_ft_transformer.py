"""Train the proposed multitask FT-Transformer on the synthetic MWCCS proxy."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NUMERICAL = [
    "AGEATVIS", "HEIGHT", "WEIGHT", "BMI", "TOTCHOL", "HDL", "TRIGLYC",
    "NONHDL", "MHLDLC", "ALBU", "PROT", "GLOBULIN", "A_G_RATIO",
    "TC_HDL_RATIO", "LDL_HDL_RATIO", "TG_HDL_RATIO", "HBA1C", "GLUCOSE",
    "CREAT", "AST", "ALT", "ART_DURATION_DAYS", "CD4N", "CD4P", "CD8N",
    "CD8P", "HSRAT",
]
CATEGORICAL = [
    "SITE", "STUDYHIST", "VISTYPE", "SEXBIRTH", "FASTING", "CHOLRX",
    "NIACIN", "FIBRATES", "DIABETES", "ONART", "ARTV", "HAARTV2", "ADH",
]


class Preprocessor:
    def fit(self, df: pd.DataFrame):
        self.medians = df[NUMERICAL].median().fillna(0.0)
        filled = df[NUMERICAL].fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1).fillna(1)
        self.maps = {}
        for col in CATEGORICAL:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            self.maps[col] = {v: i + 1 for i, v in enumerate(vals)}  # 0 = missing/unknown
        return self

    def transform(self, df: pd.DataFrame):
        xn = ((df[NUMERICAL].fillna(self.medians) - self.means) / self.stds).to_numpy(np.float32)
        xc = np.stack([
            df[c].astype("string").map(self.maps[c]).fillna(0).to_numpy(np.int64)
            for c in CATEGORICAL
        ], axis=1)
        return xn, xc

    @property
    def cardinalities(self):
        return [len(self.maps[c]) + 1 for c in CATEGORICAL]

    def state_dict(self):
        return {"numerical": NUMERICAL, "categorical": CATEGORICAL,
                "medians": self.medians.to_dict(), "means": self.means.to_dict(),
                "stds": self.stds.to_dict(), "category_maps": self.maps,
                "cardinalities": self.cardinalities}


class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities, d=64, heads=8, blocks=3, ffn=128, dropout=.2):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.empty(n_num, d))
        nn.init.xavier_uniform_(self.num_weight)
        nn.init.zeros_(self.num_bias)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(n, d) for n in cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks, norm=nn.LayerNorm(d))
        self.hiv_head = nn.Linear(d, 1)
        self.art_head = nn.Linear(d, 1)

    def forward(self, xn, xc):
        num_tokens = xn.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tokens = torch.stack([emb(xc[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cls = self.cls.expand(xn.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, num_tokens, cat_tokens], dim=1))[:, 0]
        return self.hiv_head(h).squeeze(1), self.art_head(h).squeeze(1)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dataset(df, prep):
    xn, xc = prep.transform(df)
    art_y = df.ART_UNSUPPRESSED_LABEL.fillna(0).to_numpy(np.float32)
    return TensorDataset(torch.from_numpy(xn), torch.from_numpy(xc),
                         torch.tensor(df.HIV_SCREEN_LABEL.to_numpy(np.float32)),
                         torch.tensor(art_y),
                         torch.tensor(df.ART_MONITORING_ELIGIBLE.to_numpy(np.float32)))


def loss_fn(hlog, alog, yh, ya, mask):
    hiv = nn.functional.binary_cross_entropy_with_logits(hlog, yh)
    raw_art = nn.functional.binary_cross_entropy_with_logits(alog, ya, reduction="none")
    art = (raw_art * mask).sum() / mask.sum().clamp_min(1)
    return hiv + art, hiv, art


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); rows = []
    for batch in loader:
        xn, xc, yh, ya, mask = [x.to(device) for x in batch]
        hl, al = model(xn, xc)
        rows.append((torch.sigmoid(hl).cpu(), torch.sigmoid(al).cpu(), yh.cpu(), ya.cpu(), mask.cpu()))
    return [torch.cat(x).numpy() for x in zip(*rows)]


def metrics(y, p):
    pred = (p >= .5).astype(int)
    return {"n": int(len(y)), "prevalence": float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)), "brier": float(brier_score_loss(y, p)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/MWCCS_full_5844.csv")
    ap.add_argument("--output", default="artifacts")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); seed_all(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    parts = {s: df[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ["train", "validation", "test"]}
    prep = Preprocessor().fit(parts["train"])
    # Save built-in containers rather than a __main__-scoped class instance.
    joblib.dump(prep.state_dict(), out / "preprocessor.joblib")
    loaders = {s: DataLoader(dataset(d, prep), batch_size=args.batch_size,
                             shuffle=s == "train", num_workers=0) for s, d in parts.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FTTransformer(len(NUMERICAL), prep.cardinalities).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_sum = 0.; n = 0
        for batch in loaders["train"]:
            xn, xc, yh, ya, mask = [x.to(device) for x in batch]
            opt.zero_grad(set_to_none=True); hl, al = model(xn, xc)
            loss, _, _ = loss_fn(hl, al, yh, ya, mask); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            train_sum += loss.item() * len(yh); n += len(yh)
        model.eval(); val_sum = 0.; vn = 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                xn, xc, yh, ya, mask = [x.to(device) for x in batch]
                loss, _, _ = loss_fn(*model(xn, xc), yh, ya, mask)
                val_sum += loss.item() * len(yh); vn += len(yh)
        rec = {"epoch": epoch, "train_loss": train_sum/n, "validation_loss": val_sum/vn}
        history.append(rec); print(json.dumps(rec), flush=True)
        if rec["validation_loss"] < best - 1e-5:
            best, stale = rec["validation_loss"], 0
            torch.save({"model_state": model.state_dict(), "cardinalities": prep.cardinalities,
                        "numerical": NUMERICAL, "categorical": CATEGORICAL, "config": vars(args)}, out/"best_model.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(out/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    ph, pa, yh, ya, mask = predict(model, loaders["test"], device)
    eligible = mask.astype(bool)
    result = {"device": str(device), "best_epoch": int(np.argmin([x["validation_loss"] for x in history])+1),
              "best_validation_loss": best, "hiv_screening": metrics(yh.astype(int), ph),
              "art_monitoring_eligible_only": metrics(ya[eligible].astype(int), pa[eligible]),
              "feature_note": "Direct targets/eligibility, HIVPOS, viral-load variables, IDs, dates and split fields excluded."}
    pd.DataFrame(history).to_csv(out/"training_history.csv", index=False)
    pred = parts["test"][["CCSID"]].copy()
    pred["hiv_label"], pred["hiv_probability"] = yh.astype(int), ph
    pred["art_eligible"], pred["art_label"], pred["art_probability"] = mask.astype(int), np.where(eligible, ya, np.nan), pa
    pred.to_csv(out/"test_predictions.csv", index=False)
    (out/"metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
