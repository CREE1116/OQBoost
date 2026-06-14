"""DGCS A/B: OQBoost-only, full deployment protocol (best_params + ES + cats).

Run once per dylib build (baseline vs DGCS). Prints AUC/acc/logloss/time per
dataset so the two runs can be diffed. Uses the exact production protocol from
_utils (n_estimators=1000, lr/depth from best_params, early stopping, class
weighting, categorical handling) — the protocol FINDINGS.md requires.

Usage:
    python benchmark/dgcs_ab.py            # binary real datasets
    python benchmark/dgcs_ab.py --tag DGCS # label only
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).parent))
from _utils import _make_oqboost, _encode_cats, N_REPS, TEST_SIZE, VAL_FRAC

import adult as adult_mod
import credit_default as credit_mod
import give_me_some_credit as gmsc_mod

DATASETS = [
    ("adult",               adult_mod.load_data,  getattr(adult_mod,  "CAT_IDX", None)),
    ("credit_default",      credit_mod.load_data, getattr(credit_mod, "CAT_IDX", None)),
    ("give_me_some_credit", gmsc_mod.load_data,   getattr(gmsc_mod,   "CAT_IDX", None)),
]


def run_dataset(name, load_data, cat_idx, n_reps=N_REPS):
    X, y = load_data()
    aucs, accs, lls, times = [], [], [], []
    for rep in range(n_reps):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=rep, stratify=y)
        if cat_idx:
            Xtr_e, Xte_e = _encode_cats(Xtr, Xte, cat_idx)
        else:
            Xtr_e, Xte_e = Xtr, Xte

        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr_e, ytr, test_size=VAL_FRAC, random_state=0, stratify=ytr)

        clf = _make_oqboost(cat_idx, rep, name)
        t0 = time.perf_counter()
        clf.fit(Xtr2, ytr2, eval_set=[(Xval, yval)])
        ft = time.perf_counter() - t0

        proba = clf.predict_proba(Xte_e)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        proba /= proba.sum(1, keepdims=True)
        aucs.append(roc_auc_score(yte, proba[:, 1]))
        accs.append(accuracy_score(yte, proba.argmax(1)))
        lls.append(log_loss(yte, proba))
        times.append(ft)

    return (np.mean(aucs), np.std(aucs), np.mean(accs),
            np.mean(lls), np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--reps", type=int, default=N_REPS)
    args = ap.parse_args()

    print(f"\n{'='*72}")
    print(f"  DGCS A/B  [{args.tag}]  reps={args.reps}  (production protocol)")
    print(f"{'='*72}")
    print(f"  {'Dataset':<22}  {'AUC':>16}  {'Acc':>8}  {'LogLoss':>9}  {'Time':>8}")
    for name, load_data, cat_idx in DATASETS:
        try:
            au_m, au_s, ac_m, ll_m, t_m = run_dataset(name, load_data, cat_idx, args.reps)
            print(f"  {name:<22}  {au_m:.4f}±{au_s:.4f}  {ac_m:.4f}  {ll_m:.4f}  {t_m:6.1f}s")
        except Exception as e:
            print(f"  {name:<22}  FAILED: {e}")


if __name__ == "__main__":
    main()