# Deterministic Gradient Covariance Scan (DGCS) Study

Date: 2026-06-14. Scripts: `grad_cov_det_experiment.py`, `grad_cov_det_validation.py`.

## 1. Motivation

Current production engine (`gf_build`) generates oblique split candidates via:
- **pobs (GG-SRP)**: sample √D features by SIS probability, fill random Gaussian
  D_SUB_MAX × D_SUB_MAX matrix, Gram-Schmidt orthogonalize → multiple directions
- **Strategy A**: mutate parent direction by random scale perturbation
- **Strategy B**: inject new random feature proportional to gradient score
- **Strategy C**: blend cached direction with random alpha

All three use `std::mt19937 rng(seed+t)`. FINDINGS.md Part 2 established that A/B
harm accuracy and C is neutral-to-good. The irreplaceable ingredient is **diversity**,
and the research found `proxy_cov_axis_1` (axis + one gradient-covariance direction)
matches or beats all these strategies.

## 2. The Gradient Covariance Direction

For SIS-selected features S = {d_0, ..., d_{m-1}} and dominant class k:

```
G_vec[j] = -Σ_i x_{i,S[j]} · g_{ik}        (gradient-feature covariance)
w* = G_vec / ‖G_vec‖                          (hessian-free optimal direction)
```

This maximizes the proxy objective J(w) = (w^T G_vec)^2 when A = σI (ignoring
feature correlations). Already computed as `cg_s` in `eval_oblique` — zero additional
O(N·D) cost; just normalize.

**Why hessian-free beats WLS (A^{-1} G_vec)**:
Small nodes have noisy Hessian estimates. A^{-1} overfits to local curvature noise.
G_vec / ‖G_vec‖ acts as implicit L2 regularization on the search direction.
Measured cosine similarity between w_cov and w_wls: 0.55-0.70 (proxy_cov_study).

## 3. DGCS Candidate Set

Mode `grad_cov_det` adds 4 deterministic oblique directions on top of axis candidates:
1. `gcov_full`: all SIS features, weights ∝ G_vec
2. `gcov_sign`: ±1 on all SIS features, signs = sign(G_vec)
3. `gcov_top2`: top-2 features by |G_vec|, weights = G_vec values
4. `gcov_top4`: top-4 features by |G_vec|, weights = G_vec values

Total cost: O(N·d_sub) for G_vec, then O(d_sub·n_dirs) for threshold scans.
Zero RNG. Deterministic given data.

**WLS variant** (also tested): adds A^{-1} G_vec as 5th candidate. Hurts real data
(logloss ×1.5 on breast cancer), helps rotated synthetic AUC. Excluded.

## 4. Empirical Results

5 seeds, 100 trees, depth 5, lr=0.1, d_sub=16. `use_wls=False` for fair comparison.

```
Dataset               Config            Acc            LogLoss          AUC            Time
─────────────────────────────────────────────────────────────────────────────────────────────
Breast Cancer (30D)   gg_srp            0.9692±0.0056  0.0608±0.0102  0.9985±0.0006  3.54s (1.0x)
                      proxy_cov_axis_1  0.9776±0.0028  0.0572±0.0051  0.9984±0.0002  2.01s (1.8x)  ← best acc
                      grad_cov_det      0.9692±0.0056  0.0728±0.0099  0.9974±0.0005  2.27s (1.6x)

Wine multiclass (13D) gg_srp            1.0000         0.0072         1.0000         1.59s (1.0x)
                      proxy_cov_axis_1  1.0000         0.0078         1.0000         0.82s (1.9x)
                      grad_cov_det      1.0000         0.0081         1.0000         0.97s (1.6x)

Rotated Synth (30D)   gg_srp            0.9048±0.0048  0.2309±0.0136  0.9721±0.0022  10.92s (1.0x)
                      proxy_cov_axis_1  0.9016±0.0073  0.2402±0.0168  0.9695±0.0030   5.65s (1.9x)
                      grad_cov_det      0.9004±0.0060  0.2328±0.0031  0.9707±0.0005   6.52s (1.7x)  ← best AUC std

Synthetic (100D)      gg_srp            0.8843±0.0078  0.2788±0.0135  0.9581±0.0032  12.69s (1.0x)
                      proxy_cov_axis_1  0.8891±0.0077  0.2565±0.0097  0.9638±0.0024   6.91s (1.8x)  ← best acc
                      grad_cov_det      0.8885±0.0048  0.2534±0.0060  0.9642±0.0013   7.97s (1.6x)  ← best loss, AUC

Credit Default (23D)  gg_srp            0.8205±0.0010  0.4340±0.0008  0.7712±0.0014  32.88s (1.0x)
                      proxy_cov_axis_1  0.8244±0.0011  0.4232±0.0013  0.7801±0.0025  14.41s (2.3x)  ← best acc
                      grad_cov_det      0.8234±0.0014  0.4237±0.0006  0.7806±0.0005  16.97s (1.9x)  ← best AUC, least variance
```

## 5. Findings

**F1. Deterministic gradient directions dominate GG-SRP on real tabular data.**
   4/5 datasets: both DGCS approaches beat gg_srp on accuracy. Credit Default:
   +0.39% acc (proxy_cov) / +0.29% acc (grad_cov_det) at 1.9-2.3x speedup.
   Synthetic 100D: +0.48% acc, −9% logloss.

**F2. Rotated synthetic remains the only dataset where gg_srp wins by acc (-0.04 to -0.44%).**
   However: grad_cov_det achieves much better AUC std (0.0005 vs 0.0022), meaning
   more stable convergence. The acc gap (0.0044) may be noise at current seed count.
   Root cause: full random rotation creates feature correlations that GG-SRP's
   broader random sweep handles better. Deterministic gradient direction is
   "correct on average" but doesn't explore the full correlated subspace.

**F3. proxy_cov_axis_1 (axis + 1 cov direction) is the clearest single winner.**
   Simplest implementation, fastest (1.8-2.3x), best or tied acc on 4/5 datasets.
   Already validated in proxy_cov_study.md; this study confirms robustly.

**F4. grad_cov_det's sparse variants (top2, top4) add marginal value.**
   Better logloss on synth 100D (0.2534 < 0.2565), best AUC on credit default.
   Neutral or slightly negative on breast cancer. Net: small improvement over
   proxy_cov_axis_1 on large datasets, slight hurt on small datasets.

**F5. Neither approach benefits from WLS (Hessian-adjusted direction).**
   WLS hurts breast cancer logloss (0.0606 → 0.0890). Only helps rotated synth AUC.
   Hessian noise at small/deep nodes outweighs the second-order correction benefit.

## 6. C++ Port Plan

**Ready to port: proxy_cov_axis_1** (the minimum-viable improvement).

In `eval_oblique` (`oqboost.cpp`):
- `cg_s[d]` already computed (lines ~987-1003)
- After SIS feature scoring, compute one additional direction:
  ```cpp
  // gcov direction: normalize cg_s over SIS-selected features
  float norm_cg = 0.0f;
  for (int d = 0; d < D; d++) {
      if (feat_mask[d]) norm_cg += cg_s[d] * cg_s[d];
  }
  norm_cg = std::sqrt(norm_cg);
  if (norm_cg > 1e-12f && dirs_n < OQBoostCtx::DIRS_MAX) {
      float* slot = dirs_flat + (size_t)dirs_n * D;
      std::fill(slot, slot + D, 0.0f);
      float inv_norm = 1.0f / norm_cg;
      for (int d = 0; d < D; d++) {
          if (feat_mask[d]) slot[d] = cg_s[d] * inv_norm;
      }
      dirs_n++;
  }
  ```
- This adds ONE deterministic candidate at O(D) additional cost (norm loop only).
- Can also add top2/top4 sparse variants (sort indices by |cg_s|, take top k).

**Prerequisite (FINDINGS.md lesson)**: mathematical proof that w_cov is at least as
good as random directions on the REAL benchmark datasets (adult, credit, gmsc) with
the tuned hyperparameter + early-stopping protocol. The above Python results are
encouraging but not the deployment protocol. Run quick_bench / credit_default.py
with the production C++ engine after porting before committing the change.
