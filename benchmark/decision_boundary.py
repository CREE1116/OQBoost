"""Decision boundary visualization: OQBoost vs XGBoost / LightGBM / CatBoost.

Renders a grid — rows = synthetic 2D dataset shapes (binary AND 3-class),
columns = models — so the geometry each booster carves is directly comparable.
OQBoost's oblique splits should show smoother diagonal boundaries where
axis-aligned boosters produce staircase artifacts.

Usage:
    python benchmark/decision_boundary.py
    python benchmark/decision_boundary.py --n 1500 --grid 300 --out bd.png
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

from sklearn.datasets import (make_moons, make_circles, make_classification,
                              make_blobs, make_gaussian_quantiles)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from oqboost import OQBoostClassifier


# class color palette (vivid for points, lightened for region fill)
BASE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]   # red blue green purple


def _lighten(hex_color: str, f: float = 0.55) -> tuple[float, float, float]:
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# ─── Synthetic 2D dataset shapes (binary + 3-class) ─────────────────────────────

def _spiral(n, arms, seed):
    rng = np.random.RandomState(seed)
    per = n // arms
    X, y = [], []
    for a in range(arms):
        t = np.sqrt(rng.rand(per)) * 2.7 * np.pi
        off = 2.0 * np.pi * a / arms
        x = np.stack([t * np.cos(t + off), t * np.sin(t + off)], 1)
        x += rng.normal(0, 0.35, x.shape)
        X.append(x); y.append(np.full(per, a))
    return np.vstack(X), np.concatenate(y)


def make_datasets(n: int, seed: int):
    rng = np.random.RandomState(seed)
    ds = []

    # ── binary shapes ──
    X, y = make_moons(n_samples=n, noise=0.25, random_state=seed)
    ds.append(("moons (2)", X, y))

    X, y = make_circles(n_samples=n, noise=0.12, factor=0.45, random_state=seed)
    ds.append(("circles (2)", X, y))

    X, y = make_blobs(n_samples=n, centers=2, cluster_std=1.6, random_state=seed)
    th = np.pi / 5
    X = X @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ds.append(("rotated_blobs (2)", X, y))

    X, y = make_classification(
        n_samples=n, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.0, flip_y=0.05, random_state=seed)
    X = X @ np.array([[1.0, 0.8], [0.0, 0.6]])
    ds.append(("correlated (2)", X, y))

    Xx = rng.uniform(-3, 3, size=(n, 2))
    yy = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
    Xx += rng.normal(0, 0.25, size=Xx.shape)
    ds.append(("xor (2)", Xx, yy))

    # ── 3-class shapes ──
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=1.5, random_state=seed)
    X = X @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ds.append(("blobs (3)", X, y))

    X, y = make_gaussian_quantiles(
        n_samples=n, n_features=2, n_classes=3, random_state=seed)
    ds.append(("quantiles (3)", X, y))

    X, y = _spiral(n, arms=3, seed=seed)
    ds.append(("spiral (3)", X, y))

    X, y = make_classification(
        n_samples=n, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, n_classes=3, class_sep=1.3, flip_y=0.03,
        random_state=seed)
    X = X @ np.array([[1.0, 0.7], [0.0, 0.7]])
    ds.append(("correlated (3)", X, y))

    return [(name, StandardScaler().fit_transform(X).astype(np.float32),
             y.astype(int)) for name, X, y in ds]


# ─── Models ─────────────────────────────────────────────────────────────────────

def make_models(seed: int):
    common = dict(n_estimators=200, learning_rate=0.1, random_state=seed)
    return [
        ("XGBoost", XGBClassifier(
            max_depth=4, verbosity=0, tree_method="hist", **common)),
        ("LightGBM", LGBMClassifier(
            max_depth=4, num_leaves=16, verbose=-1, **common)),
        ("CatBoost", CatBoostClassifier(
            depth=4, verbose=0, random_seed=seed,
            n_estimators=200, learning_rate=0.1)),
        ("OQBoost", OQBoostClassifier(
            max_depth=4, reg_lambda=1.0, subsample=0.8,
            early_stopping_rounds=None, **common)),
    ]


# ─── Rendering ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="samples per dataset")
    ap.add_argument("--grid", type=int, default=300, help="mesh resolution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark/results/figures/decision_boundaries.png")
    args = ap.parse_args()

    datasets = make_datasets(args.n, args.seed)
    nrows, ncols = len(datasets), 4
    plt.rcParams.update({"figure.facecolor": "white"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.9 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    rng_pt = np.random.RandomState(0)

    for r, (dname, X, y) in enumerate(datasets):
        K = int(y.max()) + 1
        bg_cmap = ListedColormap([_lighten(BASE[k]) for k in range(K)])

        x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
        y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, args.grid),
            np.linspace(y_min, y_max, args.grid))
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

        # subsample points for a clean scatter (max ~280 per class)
        sel = []
        for cls in range(K):
            ci = np.flatnonzero(y == cls)
            if len(ci) > 280:
                ci = rng_pt.choice(ci, 280, replace=False)
            sel.append(ci)
        sel = np.concatenate(sel)
        Xp, yp = X[sel], y[sel]

        for c, (mname, model) in enumerate(make_models(args.seed)):
            ax = axes[r, c]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
                pred = model.predict(grid).reshape(xx.shape)
                train_acc = (model.predict(X) == y).mean()

            ax.pcolormesh(xx, yy, pred, cmap=bg_cmap, shading="auto",
                          alpha=0.6, rasterized=True, vmin=0, vmax=K - 1)
            # crisp class-region edges (works for any K)
            ax.contour(xx, yy, pred, levels=np.arange(K - 1) + 0.5,
                       colors="#111111", linewidths=1.3)
            ax.scatter(Xp[:, 0], Xp[:, 1], c=[BASE[k] for k in yp], s=11,
                       edgecolors="white", linewidths=0.45, alpha=0.95)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#bbbbbb")
            if r == 0:
                ax.set_title(mname, fontsize=13, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(dname, fontsize=12, fontweight="bold")
            ax.text(0.96, 0.04, f"acc {train_acc:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", alpha=0.85))

    fig.suptitle("Decision Boundaries — synthetic 2D shapes (binary + 3-class) × boosters",
                 fontsize=15, fontweight="bold", y=1.002)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved → {args.out}  ({nrows}×{ncols} grid, n={args.n})")


if __name__ == "__main__":
    main()