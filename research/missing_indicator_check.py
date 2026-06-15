"""Validate missing-indicator features before C++ engine change.

Adds φ_miss=1/0 columns (one per numeric column that has NaNs) and checks
whether OQBoost's DGCS picks up Cov(missing, g) signal — i.e. whether the
indicator helps under injected missingness vs the current mean-impute path.

Same data as benchmark/missing_value_robustness.py (OpenML 42477).
"""
from __future__ import annotations

import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from oqboost import OQBoostClassifier


def add_indicators(X_tr, X_te):
    """Append missing-indicator columns for numeric cols with NaNs in train."""
    m_tr = np.isnan(X_tr)
    cols = [j for j in range(X_tr.shape[1]) if m_tr[:, j].any()]
    if not cols:
        return X_tr, X_te
    ind_tr = m_tr[:, cols].astype(np.float32)
    ind_te = np.isnan(X_te)[:, cols].astype(np.float32)
    return (np.hstack([X_tr, ind_tr]).astype(np.float32),
            np.hstack([X_te, ind_te]).astype(np.float32))


def inject(X, ratio, rng):
    X = X.copy()
    X[rng.random(X.shape) < ratio] = np.nan
    return X


def run(X, y, ratio, indicators, n_reps=3):
    aucs, accs = [], []
    K = int(y.max()) + 1
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        Xm = inject(X, ratio, rng)
        Xtr, Xte, ytr, yte = train_test_split(
            Xm, y, test_size=0.2, random_state=rep, stratify=y)
        if indicators:
            Xtr, Xte = add_indicators(Xtr, Xte)
        Xt2, Xv, yt2, yv = train_test_split(
            Xtr, ytr, test_size=0.1, random_state=0, stratify=ytr)
        clf = OQBoostClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, reg_lambda=1.0,
            subsample=0.8, early_stopping_rounds=50, cat_features=None,
            random_state=rep, verbose=False)
        clf.fit(Xt2, yt2, eval_set=[(Xv, yv)])
        proba = clf.predict_proba(Xte)
        if K == 2:
            aucs.append(roc_auc_score(yte, proba[:, 1]))
        else:
            aucs.append(roc_auc_score(yte, proba, multi_class="ovr"))
        accs.append(accuracy_score(yte, proba.argmax(1)))
    return np.mean(aucs), np.std(aucs), np.mean(accs)


def main():
    print("loading OpenML 42477 ...")
    ds = fetch_openml(data_id=42477, as_frame=False, parser="auto")
    X = ds.data.astype(np.float32)
    y = ds.target.astype(int)
    print(f"  {X.shape}, K={int(y.max())+1}\n")
    print(f"  {'ratio':>6} {'base AUC':>16} {'+indicator AUC':>18} {'Δ':>8}")
    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4]:
        b_au, b_s, b_ac = run(X, y, ratio, indicators=False)
        i_au, i_s, i_ac = run(X, y, ratio, indicators=True)
        print(f"  {ratio:>6.0%} {b_au:.4f}±{b_s:.4f}   {i_au:.4f}±{i_s:.4f}   {i_au-b_au:+.4f}")


if __name__ == "__main__":
    main()
