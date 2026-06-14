"""Why does DGCS lag GG-SRP on adult? Node-level diagnosis via PyTorch model.

Compares grad_cov_det vs gg_srp on adult (ordinal-encoded, NaN-imputed):
  - gain_ratio:  best_gain / oracle_gain per node (how close to the best
    achievable oblique split each method's winner gets)
  - winner_type distribution (axis vs oblique family wins)
  - per-depth mean gain
  - obliqueness of winners (are good adult splits axis-like?)

Hypothesis: adult signal is mostly axis-aligned (categorical + sparse skewed
numeric), so oblique gain headroom is small; GG-SRP's random pool occasionally
lands a slightly better near-axis oblique that the 4 deterministic DGCS
directions miss.
"""
from __future__ import annotations

import os, sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from research.oqboost_research import OQBoostResearch
import adult as adult_mod


def load_adult_numeric(n=6000, seed=0):
    X, y = adult_mod.load_data()
    # mean-impute NaN (research model has no native NaN handling)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    if len(Xtr) > n:
        Xtr, ytr = Xtr[:n], ytr[:n]
    return Xtr, ytr, Xte, yte


def analyze(mode, Xtr, ytr, Xte, yte, n_est=40, seed=0):
    clf = OQBoostResearch(
        n_estimators=n_est, learning_rate=0.1, max_depth=5, reg_lambda=2.0,
        d_sub=16, inherit_mode=mode, inherited_rp_ratio=1.0,
        n_random=0, n_inherit=4, use_wls=False,
        random_state=seed, device='cpu', record_alignment=True,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    auc = roc_auc_score(yte, proba[:, 1])

    recs = clf.all_split_records()
    gain_ratios = [r.gain_ratio for r in recs if r.gain_ratio is not None]
    cos_oracle  = [r.cos_oracle for r in recs if r.cos_oracle is not None]
    win = Counter(r.winner_type for r in recs)
    obliq = [r.angle_from_axis for r in recs]
    oblique_wins = sum(1 for r in recs if r.is_oblique_winner)

    return {
        'auc': auc,
        'n_splits': len(recs),
        'gain_ratio_mean': float(np.mean(gain_ratios)) if gain_ratios else float('nan'),
        'gain_ratio_med': float(np.median(gain_ratios)) if gain_ratios else float('nan'),
        'cos_oracle_mean': float(np.mean(cos_oracle)) if cos_oracle else float('nan'),
        'obliqueness_mean': float(np.mean(obliq)),
        'oblique_win_rate': oblique_wins / max(len(recs), 1),
        'winners': dict(win),
    }


def main():
    print("=" * 70)
    print("  Adult diagnosis: grad_cov_det vs gg_srp (PyTorch research model)")
    print("=" * 70)
    Xtr, ytr, Xte, yte = load_adult_numeric()
    print(f"  data: {Xtr.shape[0]} train / {Xte.shape[0]} test / {Xtr.shape[1]} feats (ordinal+imputed)\n")

    for mode in ['gg_srp', 'grad_cov_det']:
        r = analyze(mode, Xtr, ytr, Xte, yte)
        print(f"[{mode}]")
        print(f"  test AUC          : {r['auc']:.4f}")
        print(f"  splits            : {r['n_splits']}")
        print(f"  gain_ratio (mean) : {r['gain_ratio_mean']:.4f}  (best_gain / oracle_gain)")
        print(f"  gain_ratio (med)  : {r['gain_ratio_med']:.4f}")
        print(f"  cos(winner,oracle): {r['cos_oracle_mean']:.4f}")
        print(f"  obliqueness (deg) : {r['obliqueness_mean']:.1f}")
        print(f"  oblique win rate  : {r['oblique_win_rate']*100:.1f}%")
        print(f"  winner types      : {r['winners']}")
        print()


if __name__ == '__main__':
    main()
