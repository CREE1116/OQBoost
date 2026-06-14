"""
Extended validation: DGCS vs GG-SRP across more datasets.

Adds: credit_default (23D), wine (13D), high-dim synthetic (100D).
5 seeds for tighter confidence intervals.
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch

from sklearn.datasets import (load_breast_cancer, load_wine, make_classification)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

try:
    from sklearn.datasets import fetch_openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False


CONFIGS = [
    ("gg_srp",           "gg_srp",           16),
    ("proxy_cov_axis_1", "proxy_cov_axis_1", 16),
    ("grad_cov_det",     "grad_cov_det",     16),
]

N_SEEDS   = 5
N_TREES   = 100
MAX_DEPTH = 5
LR        = 0.1


def run(inherit_mode, d_sub, X_tr, y_tr, X_te, y_te, n_seeds):
    accs, losses, aucs, times = [], [], [], []
    for seed in range(n_seeds):
        t0 = time.time()
        clf = OQBoostResearch(
            n_estimators=N_TREES, learning_rate=LR, max_depth=MAX_DEPTH,
            reg_lambda=1.0, d_sub=d_sub,
            inherited_rp_ratio=0.0, n_random=0, n_inherit=0,
            inherit_mode=inherit_mode, use_wls=False,
            random_state=seed, device='cpu',
        )
        clf.fit(X_tr, y_tr)
        times.append(time.time() - t0)
        proba = clf.predict_proba(X_te)
        pred  = proba.argmax(1)
        accs.append(accuracy_score(y_te, pred))
        losses.append(log_loss(y_te, proba))
        K = proba.shape[1]
        try:
            aucs.append(roc_auc_score(y_te, proba[:, 1] if K == 2 else proba,
                                      multi_class='ovr' if K > 2 else 'raise'))
        except Exception:
            aucs.append(float('nan'))
    return (np.mean(accs), np.std(accs),
            np.mean(losses), np.std(losses),
            np.mean(aucs), np.std(aucs),
            np.mean(times))


def report(ds, X_tr, y_tr, X_te, y_te):
    print(f"\n{ds}  [{X_tr.shape[0]} tr / {X_te.shape[0]} te / {X_tr.shape[1]}D]")
    print(f"  {'Config':<22}  {'Acc':>12}  {'LogLoss':>12}  {'AUC':>12}  {'Time':>7}")
    baseline_time = None
    for name, mode, dsub in CONFIGS:
        a_m, a_s, l_m, l_s, au_m, au_s, t = run(mode, dsub, X_tr, y_tr, X_te, y_te, N_SEEDS)
        if baseline_time is None:
            baseline_time = t
        sp = baseline_time / (t + 1e-9)
        print(f"  {name:<22}  {a_m:.4f}±{a_s:.4f}  {l_m:.4f}±{l_s:.4f}  "
              f"{au_m:.4f}±{au_s:.4f}  {t:.2f}s ({sp:.1f}x)")


def sc(X):
    return StandardScaler().fit_transform(X).astype(np.float32)


def main():
    print("=" * 72)
    print("  DGCS Extended Validation  |  5 seeds, 100 trees, depth 5")
    print("=" * 72)

    # Breast cancer
    d = load_breast_cancer()
    X, y = sc(d.data), d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    report("Breast Cancer (30D)", Xtr, ytr, Xte, yte)

    # Wine (multiclass)
    d = load_wine()
    X, y = sc(d.data), d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    report("Wine multiclass (13D)", Xtr, ytr, Xte, yte)

    # Rotated synthetic 30D
    X, y = make_classification(n_samples=2000, n_features=30, n_informative=10,
                                n_redundant=5, random_state=0)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    X = (X @ Q).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    report("Rotated Synthetic (30D)", Xtr, ytr, Xte, yte)

    # High-dim synthetic 100D
    X, y = make_classification(n_samples=3000, n_features=100, n_informative=20,
                                n_redundant=10, random_state=0)
    X = sc(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    report("Synthetic (100D)", Xtr, ytr, Xte, yte)

    # Credit default (OpenML) — tabular real data
    if HAS_OPENML:
        try:
            d = fetch_openml("default-of-credit-card-clients", version=1,
                             as_frame=False, parser="liac-arff")
            X = d.data.astype(np.float32)
            y = d.target.astype(int)
            mask = ~np.isnan(X).any(1)
            X, y = X[mask], y[mask]
            X = sc(X)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
            report("Credit Default (23D)", Xtr, ytr, Xte, yte)
        except Exception as e:
            print(f"\n[credit default fetch failed: {e}]")


if __name__ == '__main__':
    main()
