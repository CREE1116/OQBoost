# OQBoost: Algorithm and Theory

OQBoost is a gradient-boosted **oblique** decision tree ensemble. Its defining
idea is a single, fully deterministic principle for generating split directions
— the **Deterministic Gradient Covariance Scan (DGCS)** — under which continuous,
categorical, and missing values are all handled by one covariance framework.

This document describes the current engine. (Earlier releases used a stochastic
candidate tournament — GG-SRP random projections, parent mutation, a direction
cache. Those were removed in favor of DGCS; see §7.)

---

## 1. Newton-Raphson Boosting Framework

Given the ensemble prediction $F^{(m)}(x)$ at round $m$, the next tree minimizes
the second-order Taylor expansion of the loss:

$$\mathcal{L}^{(m+1)} \approx \sum_{i=1}^{N} \left[ g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2 \right] + \tfrac{1}{2}\lambda \sum_{\text{leaves}} w_\text{leaf}^2$$

* $g_i$ — first-order gradient, $h_i$ — second-order Hessian, both at $F^{(m)}$.
* Leaf value (Newton step): $w_\text{leaf} = -\,\dfrac{\sum_i g_i}{\sum_i h_i + \lambda}$.
* $K$-class classification keeps $K$ output heads under a softmax loss.

---

## 2. Oblique Splits

Each internal node splits on a linear combination of features rather than a
single feature:

$$\sum_{j \in \mathcal{S}} w_j\,\phi_j(x) < \theta$$

with a sparse weight vector $w$ ($|\mathcal{S}| \le D_\text{SUB\_MAX}=16$) and a
threshold $\theta$. $\phi_j$ is the **unified feature embedding** of §4. Oblique
boundaries capture rotated / correlated structure that axis-aligned trees can
only approximate with staircases.

---

## 3. DGCS — Deterministic Gradient Covariance Scan

The central question at a node is: *which direction $w$ best separates the
gradient?* The second-order split-gain proxy is

$$J(w) = \frac{(w^\top G)^2}{w^\top H w + \lambda\lVert w\rVert^2}, \qquad G_j = \textstyle\sum_{i} \phi_j(x_i)\, g_{ik}$$

where $G$ is the **gradient–feature covariance** and $k$ the dominant class. DGCS
does not search for the maximizer — it writes it down in closed form.

### 3.1 Diagonal-Newton direction

Approximating $H$ by its diagonal $\operatorname{diag}(\sum_i h_i \phi_j^2)$ — call
it $A_j$ — the maximizer is

$$w^*_j \;\propto\; \frac{G_j}{A_j + \lambda}.$$

This is the per-feature Newton response. The diagonal form is **scale-invariant**:
the plain $H\approx\sigma I$ form ($w \propto G$) is hijacked by large-scale raw
features (e.g. a feature on a $10^5$ scale dominates the covariance), collapsing
oblique gain on unstandardized data. Dividing by $A_j$ removes that. The exact
WLS solution $w = A^{-1}G$ was tested and rejected — it overfits the noisy
small-node Hessian; the diagonal form is an implicit regularizer.

### 3.2 Sparsity variants

DGCS emits a small fixed set of deterministic candidates from the top
$d_\text{sub}$ features (ranked by the SIS score $s_j = |G_j|/\sqrt{A_j+\lambda}$):

| candidate | description |
| :-- | :-- |
| `full`  | diagonal-Newton direction over all top features |
| `sign`  | $\pm1$ gradient-sign pattern (scale-free) |
| `top2`  | sparse: 2 strongest features |
| `top4`  | sparse: 4 strongest features |

For multiclass, one covariance direction is emitted **per class**
($G^{(c)}_j = \sum_i \phi_j x_i\, g_{ic}$) so each class’s gradient geometry is
represented rather than only the dominant class.

### 3.3 Fully deterministic

No RNG anywhere in candidate generation. Given the same data and gradients, the
same directions are produced every time — reproducible, and removing the
hyperparameters the old stochastic pool needed. The only two effective
hyperparameters are `max_depth` and `reg_lambda`.

---

## 4. Unified Feature Framework

Every feature type maps to a single embedding $\phi_j$, after which DGCS sees
only $\operatorname{Cov}(\phi_j, g)$ — it cannot tell the types apart:

| type | embedding $\phi_j(v)$ |
| :-- | :-- |
| continuous   | $\phi = v$ |
| categorical  | $\phi(c) = \dfrac{G_c}{H_c+\lambda}$ — gradient-rank Newton response, recomputed **every round** |
| categorical missing | $\phi(\text{NaN}) = \dfrac{G_\text{miss}}{H_\text{miss}+\lambda}$ — its own category |
| numeric missing | $\phi(\text{NaN}) = \dfrac{G_\text{miss}}{H_\text{miss}+\lambda}$ — same Newton-response form |

A category value becomes a continuous coordinate equal to its node-adaptive
gradient response. Unlike CatBoost's static target encoding $E[y\mid c]$, this is
$E[g\mid c]$ — it adapts to the **current boosting residual** each round. Numeric
missing is treated as a one-extra "Observed/Missing" state with the same
Newton-response embedding, so the missingness signal enters the covariance
($\operatorname{Cov}(\text{missing}, g)$) when it correlates with the gradient.
The result: continuous, categorical, and missing flow through one covariance
pipeline.

---

## 5. Node Search and Tree Growth

At each node OQBoost evaluates two candidate families and keeps the best by gain:

1. **Axis scan** — a full histogram scan over all features (256 bins), the
   standard XGBoost/LightGBM-style split, including a missing-value bin with a
   learned default direction (NaNs sent left/right by gain). This is essentially
   free thanks to histogram subtraction (a child histogram is the parent minus
   its sibling) and is more precise on categorical splits than the oblique path.
2. **Oblique DGCS** — the deterministic covariance directions of §3, evaluated on
   a contiguous sample panel with a batched (BLAS GEMM) projection.

Growth is **lazy best-first**: a priority queue expands the node with the largest
potential gain first, under a leaf budget. Leaf values use the Newton step with
parent-shrinkage smoothing.

---

## 6. Memory-Optimized C++ Engine

* **Histogram object pool** — 256-bin feature histograms are recycled across
  best-first node growth, driving steady-state heap allocation to zero.
* **Histogram subtraction** — the smaller child is built fresh; the larger child
  is `parent − smaller`.
* **Dynamic hybrid parallelism** — block-wise feature-parallel histograms when
  $D \ge T$ (zero merge overhead, cache-contiguous), sample-parallel with a
  thread-local merge when $D < T$ (small buffers stay L1/L2-resident).
* **Batched oblique projection** — all candidate directions projected in one BLAS
  GEMM over a contiguous panel.
* **K=2 fast path** — binary / one-vs-rest histograms are unrolled (stride 5).

---

## 7. Why DGCS Replaced the Stochastic Pool

Earlier engines generated directions by a tournament over random families
(GG-SRP sparse random projections, parent-direction mutation, a cross-tree
direction cache). DGCS replaces all of them with the closed-form covariance
direction because:

* **It is the optimum, not a sample of it.** The stochastic pool searched for a
  good direction by drawing many candidates and keeping the best; DGCS derives
  the second-order-optimal direction directly. No search, no RNG, fewer
  hyperparameters.
* **Variance control by structure, not luck.** The covariance estimate is noisy
  at small nodes; the diagonal-Newton normalization and the SIS feature
  restriction regularize it deterministically.
* **One framework for all feature types** (§4), which the random pool could not
  provide for categoricals/missing.

A gain-weighted blend of candidates and a cross-tree direction cache were both
evaluated as variance-reduction add-ons; on real tabular data their effect was
within noise of plain DGCS, so they were removed to keep the engine simple and
light.

---

## 8. Algorithmic Complexity

| Phase | Time Complexity | Notes |
| :--- | :--- | :--- |
| Context creation | $O(N \cdot D)$ | once; bin thresholds + embeddings |
| Categorical / missing re-encoding | $O(N \cdot D)$ | once per round (gradient-adaptive) |
| DGCS direction | $O(n_t \cdot d_\text{sub})$ | per node; covariance + diagonal |
| Histogram construction | $O(n_t \cdot D \cdot 256)$ | reduced by histogram subtraction |
| Inference routing | $O(N \cdot \text{depth} \cdot s)$ | $s \le 16$ active features per split |

---

[한국어 버전 (Korean Version)](algorithm.ko.md)
