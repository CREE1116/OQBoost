# OQBoost

**High-performance gradient-boosted oblique decision trees with deterministic Gradient-Covariance Scan.**

OQBoost replaces standard axis-aligned splits with gradient-guided oblique hyperplanes computed directly via a deterministic Gradient-Covariance Scan (DGCS). It builds oblique splits without expensive numerical optimization or random search, yielding superior boundaries on complex tabular datasets.

---

## Installation

```bash
pip install oqboost
```

Pre-compiled wheels are available for macOS (arm64, x86_64) and Linux (x86_64). On other platforms, a C++17 compiler (such as `clang++` or `g++`) is required to compile from source.

---

## Quickstart

### 1. Classification (`OQBoostClassifier`)

`OQBoostClassifier` provides binary and multiclass classification. For multiclass classification, the default strategy is `"shared"` (Multi-value Shared Leaves) which trains extremely fast by building a single tree ensemble with vector-valued leaf outputs.

```python
from oqboost import OQBoostClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

clf = OQBoostClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    multi_strategy="shared",  # "shared" (fastest, default) or "ovr" (One-vs-Rest)
    random_state=42
)
clf.fit(X, y)
preds = clf.predict(X)
probas = clf.predict_proba(X)
```

### 2. Regression (`OQBoostRegressor`)

`OQBoostRegressor` supports continuous target prediction with standard GBDT loss functions.

```python
from oqboost import OQBoostRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

reg = OQBoostRegressor(
    loss="squared_error",  # "squared_error" (MSE), "absolute_error" (MAE), or "huber"
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)
reg.fit(X, y)
predictions = reg.predict(X)
```

---

## Detailed Hyperparameter Reference

### Core Parameters

* **`n_estimators`** (`int`, default=`1000`):
  * **Values**: Positive integer.
  * **Role**: The number of boosting rounds (trees to build). Increasing this generally increases model capacity, but requires early stopping to prevent overfitting.
* **`learning_rate`** (`float`, default=`0.03`):
  * **Values**: Positive float (typically in `[0.01, 0.2]`).
  * **Role**: Step size shrinkage applied to each tree's updates to prevent overfitting. Smaller values require more `n_estimators`.
* **`max_depth`** (`int`, default=`6`):
  * **Values**: Positive integer (typically in `[3, 10]`).
  * **Role**: Maximum depth of each decision tree. Allocates up to $2^{\text{max\_depth}}$ leaves per tree using a best-first (leaf-wise) strategy.
* **`max_leaves`** (`int` or `None`, default=`None`):
  * **Values**: Positive integer or `None`.
  * **Role**: Explicit leaf budget per tree. If `None`, defaults to $2^{\text{max\_depth}}$.

### Regularization

* **`reg_alpha`** (`float`, default=`0.0`):
  * **Values**: Non-negative float.
  * **Role**: L1 regularization coefficient. Applies soft-thresholding to leaf weights, shrinking small coefficients to exactly zero (encouraging tree sparsity).
* **`reg_lambda`** (`float`, default=`1.0`):
  * **Values**: Non-negative float.
  * **Role**: L2 regularization coefficient on leaf weights and split gains. Stabilizes tree weights under small samples.
* **`gamma`** (`float`, default=`0.0`):
  * **Values**: Non-negative float.
  * **Role**: Minimum split gain threshold. A node will not be split if the best split gain is less than `gamma`.
* **`min_child_weight`** (`float`, default=`1.0`):
  * **Values**: Positive float.
  * **Role**: Minimum sum of instance Hessian (data density) required in a child node. If a split creates a node with less than `min_child_weight`, the split is discarded.

### Subsampling & Speedups

* **`goss`** (`bool`, default=`False`):
  * **Values**: `True` or `False`.
  * **Role**: Activates Gradient-based One-Side Sampling (GOSS). When active, keeps samples with large gradients and randomly samples a fraction of samples with small gradients, speeding up training on large datasets by 2x+ with negligible performance loss.
* **`goss_top_rate`** (`float`, default=`0.2`):
  * **Values**: Float in `(0.0, 1.0]`.
  * **Role**: Fraction of high-gradient (large error) samples retained by GOSS.
* **`goss_other_rate`** (`float`, default=`0.1`):
  * **Values**: Float in `(0.0, 1.0]`.
  * **Role**: Fraction of low-gradient (small error) samples randomly sampled by GOSS.
* **`subsample`** (`float`, default=`0.8`):
  * **Values**: Float in `(0.0, 1.0]`.
  * **Role**: Row subsampling ratio used to build trees (ignored if `goss=True`).
* **`colsample_bynode`** (`float`, default=`1.0`):
  * **Values**: Float in `(0.0, 1.0]`.
  * **Role**: Feature subsampling ratio. Evaluates only a subset of features at each node split candidate.
* **`max_bin`** (`int`, default=`255`):
  * **Values**: Integer in `[2, 255]`.
  * **Role**: Maximum number of buckets for continuous values. Lower values (like `63` or `31`) speed up training dramatically by increasing CPU cache efficiency.

### Loss & Strategy Configurations

* **`loss`** (`str`, default=`"squared_error"`, **Regressor Only**):
  * **Values**: `"squared_error"` (MSE), `"absolute_error"` (MAE), `"huber"`.
  * **Role**: The regression objective function to minimize.
* **`huber_delta`** (`float`, default=`1.0`, **Regressor Only**):
  * **Values**: Positive float.
  * **Role**: Delta threshold at which Huber loss switches from quadratic (MSE) to linear (MAE).
* **`multi_strategy`** (`str`, default=`"shared"`, **Classifier Only**):
  * **Values**: `"shared"` or `"ovr"`.
  * **Role**: Multiclass strategy. `"shared"` builds a single tree ensemble with multi-value vector leaf outputs (fastest). `"ovr"` trains separate binary trees per class (One-vs-Rest, standard GBDT parity).

---

## Features Usage

### Native NaN & Categorical Support
```python
import numpy as np
import pandas as pd

# NaNs are handled natively during split sweeps
X_train[50, 3] = np.nan

# Categoricals are rank-encoded based on gradient ranks per round
clf = OQBoostClassifier(cat_features=["city", "product"])
clf.fit(X_train, y_train)
```

### Early Stopping & Serialization
```python
from oqboost import load_model

# Early stopping
clf = OQBoostClassifier(n_estimators=2000, early_stopping_rounds=50)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Save model
clf.save("model.joblib")

# Load model (automatically loads Classifier or Regressor)
model = load_model("model.joblib")
```

## Performance & Memory Optimizations

OQBoost is designed for high-throughput training and inference on large-scale tabular datasets. The core C++ engine and Python wrappers include several low-level optimizations:

* **Zero-Allocation Oblique Search**: Pre-allocates all oblique directions, scratch arrays, and candidate buffers (e.g., `dirs_buf`, `samp_e_buf`, `scratch_cg_s`) per binning context. This avoids expensive heap malloc/free cycles in the hot leaf-growth loop.
* **Stack-Allocated Multiclass Buffers**: Uses fast stack-allocated buffers for multiclass gradients and NaN-routing states up to `K_MAX_STACK=64` classes, bypassing heap memory traffic during multi-threaded OpenMP scans.
* **Logistic Symmetry & Softmax Cache**: Dedicated fast path for `K=2` (binary logloss) to avoid dual exponential evaluations. For `K > 2` multiclass, exp values are cached in a stack buffer to minimize costly transcendental functions.
* **Zero-Copy Purely Numerical Routing**: In `gf_predict`, if the features are purely numerical, the routing maps directly to input pointers, bypassing intermediate data copy/imputation allocation entirely.
* **In-place Python Gradients**: Gradient, Hessian, and Huber loss updates are executed via in-place NumPy functions (`out=`, `np.clip`) to prevent temporary array allocation overhead.
* **Active Categorical Cache**: Caches category index resolutions in Python (`_cat_idx_cache_`) to skip redundant $O(D)$ checks on every boosting round.
* **Active Memory Capping**: Rehearses and clears node index lists (`node_samp`) immediately after splitting, capping peak memory to $O(N \times \text{current\_depth})$ instead of holding indices for all tree levels.

---

## License

OQBoost is licensed under the MIT License. See [LICENSE](LICENSE) for details.
