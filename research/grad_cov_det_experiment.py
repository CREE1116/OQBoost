"""
Deterministic Gradient Covariance Scan (DGCS) experiment.

Compares:
  gg_srp          — production baseline: axis(d_sub) + wls(1) + random(16)
  proxy_cov_axis  — best from proxy_cov study: axis(d_sub) + 1 cov direction
  grad_cov_det    — NEW: axis(d_sub) + gcov_full + gcov_sign + gcov_top2 + gcov_top4

Key questions:
  1. Does grad_cov_det match or beat gg_srp accuracy with fewer/no random draws?
  2. Does adding sparse cov variants (top2, top4) over proxy_cov_axis_1 help?
  3. What is the speed advantage?

Datasets: rotated synthetic (30D), breast cancer (30D), adult (14D from OpenML).
3 seeds each. 100 trees, depth 5, lr 0.1.
"""
from __future__ import annotations

import os
import sys
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

try:
    from sklearn.datasets import fetch_openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def make_rotated_synthetic(n=2000, d=30, seed=0):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=10, n_redundant=5,
        n_clusters_per_class=2, random_state=seed,
    )
    # Random rotation to force oblique boundaries
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = X @ Q
    return train_test_split(X, y, test_size=0.25, random_state=seed)


def make_breast():
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)


def make_adult():
    if not HAS_OPENML:
        return None
    try:
        data = fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
        X = data.data.astype(np.float32)
        y = (data.target == ">50K").astype(int)
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"  [adult fetch failed: {e}]")
        return None


# ─── Runner ──────────────────────────────────────────────────────────────────

CONFIGS = [
    # name                inherit_mode        n_random n_inherit d_sub
    ("gg_srp (baseline)", "gg_srp",           0,       0,        16),
    ("proxy_cov_axis_1",  "proxy_cov_axis_1", 0,       0,        16),
    ("grad_cov_det",      "grad_cov_det",     0,       0,        16),
    # DGCS with WLS variant (for comparison; expected to hurt real data)
    # ("grad_cov_det+wls",  "grad_cov_det_wls", 0,       0,        16),
]

N_SEEDS = 3
N_TREES = 100
MAX_DEPTH = 5
LR = 0.1


def run_config(name, inherit_mode, n_random, n_inherit, d_sub,
               X_tr, y_tr, X_te, y_te, n_seeds=N_SEEDS):
    accs, losses, aucs, times = [], [], [], []
    for seed in range(n_seeds):
        t0 = time.time()
        clf = OQBoostResearch(
            n_estimators=N_TREES,
            learning_rate=LR,
            max_depth=MAX_DEPTH,
            reg_lambda=1.0,
            d_sub=d_sub,
            n_random=n_random,
            n_inherit=n_inherit,
            inherited_rp_ratio=(1.0 if n_inherit > 0 else 0.0),
            inherit_mode=inherit_mode,
            use_wls=False,
            random_state=seed,
            device='cpu',
        )
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        times.append(elapsed)

        proba = clf.predict_proba(X_te)
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y_te, pred))
        losses.append(log_loss(y_te, proba))
        K = proba.shape[1]
        try:
            auc = roc_auc_score(y_te, proba[:, 1] if K == 2 else proba,
                                multi_class='ovr' if K > 2 else 'raise')
        except Exception:
            auc = float('nan')
        aucs.append(auc)

    return {
        'acc':  (np.mean(accs),  np.std(accs)),
        'loss': (np.mean(losses), np.std(losses)),
        'auc':  (np.mean(aucs),  np.std(aucs)),
        'time': (np.mean(times), np.std(times)),
    }


def print_results(dataset_name, results: dict):
    print(f"\n{'─'*70}")
    print(f"  {dataset_name}")
    print(f"{'─'*70}")
    print(f"  {'Config':<22}  {'Acc':>12}  {'LogLoss':>12}  {'AUC':>12}  {'Time':>8}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}")
    for name, r in results.items():
        acc_s  = f"{r['acc'][0]:.4f}±{r['acc'][1]:.4f}"
        loss_s = f"{r['loss'][0]:.4f}±{r['loss'][1]:.4f}"
        auc_s  = f"{r['auc'][0]:.4f}±{r['auc'][1]:.4f}"
        time_s = f"{r['time'][0]:.2f}s"
        print(f"  {name:<22}  {acc_s:>12}  {loss_s:>12}  {auc_s:>12}  {time_s:>8}")


def run_dataset(ds_name, X_tr, y_tr, X_te, y_te):
    print(f"\n[{ds_name}] {X_tr.shape[0]} train / {X_te.shape[0]} test / {X_tr.shape[1]} feats")
    results = {}
    for (name, inherit_mode, n_random, n_inherit, d_sub) in CONFIGS:
        print(f"  running {name}...", end='', flush=True)
        r = run_config(name, inherit_mode, n_random, n_inherit, d_sub,
                       X_tr, y_tr, X_te, y_te)
        print(f"  acc={r['acc'][0]:.4f}  loss={r['loss'][0]:.4f}  time={r['time'][0]:.2f}s")
        results[name] = r
    print_results(ds_name, results)
    return results


def main():
    print("=" * 70)
    print("  DGCS Experiment — Deterministic Gradient Covariance Scan")
    print(f"  {N_SEEDS} seeds, {N_TREES} trees, depth {MAX_DEPTH}, lr {LR}")
    print("=" * 70)

    all_results = {}

    # 1. Rotated synthetic
    print("\n[building rotated synthetic...]")
    X_tr, X_te, y_tr, y_te = make_rotated_synthetic(n=2000, d=30, seed=0)
    X_tr = StandardScaler().fit_transform(X_tr).astype(np.float32)
    X_te = StandardScaler().fit_transform(X_te).astype(np.float32)
    all_results['rotated_synthetic (30D)'] = run_dataset(
        'Rotated Synthetic (30D)', X_tr, y_tr, X_te, y_te)

    # 2. Breast cancer
    print("\n[loading breast cancer...]")
    X_tr2, X_te2, y_tr2, y_te2 = make_breast()
    all_results['breast_cancer (30D)'] = run_dataset(
        'Breast Cancer (30D)', X_tr2, y_tr2, X_te2, y_te2)

    # 3. Adult (optional)
    print("\n[loading adult (OpenML)...]")
    adult = make_adult()
    if adult is not None:
        X_tr3, X_te3, y_tr3, y_te3 = adult
        all_results['adult'] = run_dataset('Adult', X_tr3, y_tr3, X_te3, y_te3)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — grad_cov_det vs gg_srp vs proxy_cov_axis_1")
    print("=" * 70)
    for ds, res in all_results.items():
        base_acc = res.get('gg_srp (baseline)', {}).get('acc', (float('nan'),))[0]
        for name, r in res.items():
            delta = r['acc'][0] - base_acc
            speedup = res.get('gg_srp (baseline)', {}).get('time', (1.0,))[0] / (r['time'][0] + 1e-9)
            print(f"  {ds:30s}  {name:<22}  acc={r['acc'][0]:.4f}  Δ={delta:+.4f}  {speedup:.1f}x speedup")


if __name__ == '__main__':
    main()
