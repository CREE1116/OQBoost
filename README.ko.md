# OQBoost

**직접 경사-공분산 스캔을 기반으로 한 경사 부스팅 사선 결정 트리 (Gradient-boosted oblique decision trees with deterministic Gradient-Covariance Scan)**

OQBoost는 기존의 축 정렬 분기(axis-aligned splits) 대신 그라디언트(gradient)에 의해 정렬되는 사선 하이퍼플레인(oblique hyperplanes) 분기를 수행합니다. 무작위 사영이나 유전적 변이 없이 결정론적인 직접 경사-공분산 스캔(DGCS) 방식을 통해 닫힌 형식(closed-form)으로 최적의 분기 방향을 직접 도출하며, 값비싼 수치 최적화 없이도 데이터의 고유한 기하학적 구조를 효과적으로 캡처합니다.

<p align="center">
  <img src="https://raw.githubusercontent.com/cree1116/oqboost/main/docs/diverse_boundaries.png" alt="OQBoost 결정 경계" width="800">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/cree1116/oqboost/actions/workflows/ci.yml/badge.svg)](https://github.com/cree1116/oqboost/actions)
[![PyPI](https://img.shields.io/pypi/v/oqboost)](https://pypi.org/project/oqboost/)

---

## 핵심 특징 (Key Properties)

| 특징 | 설명 |
|------|------|
| **분기 타입 (Split type)** | 사선 분기 (Oblique split, 여러 피처들의 선형 사영 결합) |
| **방향 탐색 (Direction finding)** | GG-SRP: 그라디언트 유도형 희소 무작위 사영 |
| **상속 구조 (Inheritance)** | 부모 노드의 분기 방향 가중치를 상속하고 깊이에 비례해 감쇠하는 변이 적용 |
| **결측치 지원 (Missing values)** | 네이티브 지원 — C++ 내부에서 실시간 평균 대체로 NaN 처리 |
| **범주형 피처 지원 (Categorical features)** | 네이티브 지원 — 라운드별 그라디언트 랭크(gradient-rank) 타겟 인코딩 |
| **지원 작업 (Tasks)** | 분류 (`OQBoostClassifier`) + 회귀 (`OQBoostRegressor`) |
| **API 호환성** | scikit-learn API 완벽 호환 |
| **백엔드 엔진** | OpenMP 병렬 컴파일된 C++ 백엔드 |

---

## 설치 (Install)

```bash
pip install oqboost
```

macOS (arm64, x86_64) 및 Linux (x86_64) 환경을 위한 사전 빌드된 휠(wheel) 패키지가 배포됩니다.
지원되지 않는 플랫폼의 경우, 소스코드 컴파일을 위해 `clang++` 또는 `g++`가 필요합니다.

---

## 퀵스타트 (Quickstart)

```python
from oqboost import OQBoostClassifier

clf = OQBoostClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    random_state=42,
)

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
clf.predict(X_test)
clf.predict_proba(X_test)
```

### 네이티브 결측치(NaN) 지원

```python
import numpy as np
X_train[50, 3] = np.nan   # 데이터의 임의 영역에 NaN 존재 허용
clf.fit(X_train, y_train)
clf.predict(X_test)        # 추론 시점에 컬럼 평균 대체 방식 적용
```

### 네이티브 범주형 피처 지원

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 방법 1: pandas의 Categorical 또는 object 컬럼 자동 감지
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# 방법 2: 명시적으로 범주형 컬럼 지정
clf = OQBoostClassifier(cat_features=["city", "product"])
clf.fit(X, y)
```

### 조기 종료 (Early Stopping)

```python
clf = OQBoostClassifier(n_estimators=2000, early_stopping_rounds=50)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"조기 종료 지점: {clf.get_n_trees()} 번째 트리")
```

### 회귀 (Regression)

```python
from oqboost import OQBoostRegressor

reg = OQBoostRegressor(
    loss="squared_error",   # 또는 "huber" (huber_delta 지정)
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
)
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
reg.predict(X_test)
```

### 모델 저장 및 로드 (Save / Load)

```python
clf.save("model.joblib")
from oqboost import load_model
clf2 = load_model("model.joblib")
```

---

## 벤치마크 결과 (Benchmark Results)

모든 벤치마크는 80/20 학습-평가 스플릿으로 3회 반복 실행하여 평균 및 표준편차(mean ± std)를 계산했습니다. 비교군 모델(XGBoost, LightGBM, CatBoost, OQBoost)의 하이퍼파라미터는 모두 동일하게 **Optuna를 사용하여 50회(trials) 튜닝**된 최적 값을 적용했습니다.

### 메인 벤치마크 테이블 (Main Benchmark Table)

| 데이터셋 | 모델 | 정확도 (Accuracy) | 균형 정확도 (Bal. Acc.) | F1 매크로 (F1 Macro) | 로그 손실 (Log Loss) | 학습 시간 (Train, 초) | 추론 시간 (Infer, 초) |
|---------|-------|----------|-----------|----------|----------|-----------|-----------|
| **Adult** | XGBoost | 0.8736±0.0033 | 0.7990±0.0065 | 0.8159±0.0055 | 0.2760±0.0018 | **0.83±0.10** | **0.01±0.00** |
| | LightGBM | **0.8746±0.0041** | **0.8002±0.0079** | **0.8173±0.0069** | 0.2755±0.0021 | 2.81±0.26 | 0.06±0.00 |
| | CatBoost | 0.8745±0.0040 | 0.7963±0.0065 | 0.8156±0.0063 | **0.2737±0.0018** | 7.99±0.51 | 0.05±0.00 |
| | OQBoost | 0.8712±0.0025 | 0.7991±0.0083 | 0.8139±0.0056 | 0.2804±0.0026 | 2.45±0.60 | 0.02±0.01 |
| **Credit Default** | XGBoost | 0.8206±0.0018 | 0.6584±0.0015 | 0.6836±0.0022 | 0.4280±0.0010 | **0.19±0.01** | **0.00±0.00** |
| | LightGBM | **0.8223±0.0019** | 0.6585±0.0018 | 0.6844±0.0025 | **0.4239±0.0012** | 5.31±0.65 | 0.07±0.01 |
| | CatBoost | 0.8220±0.0010 | **0.6603±0.0021** | **0.6859±0.0024** | 0.4274±0.0004 | 0.56±0.07 | **0.00±0.00** |
| | OQBoost | 0.8221±0.0022 | 0.6597±0.0033 | 0.6855±0.0039 | 0.4269±0.0004 | 0.52±0.04 | **0.00±0.00** |
| **Give Me Credit** | XGBoost | 0.7400±0.0312 | 0.6381±0.0349 | 0.6488±0.0398 | 0.5093±0.0266 | 0.15±0.02 | **0.00±0.00** |
| | LightGBM | 0.7417±0.0321 | 0.6250±0.0360 | 0.6341±0.0420 | 0.5051±0.0320 | 0.39±0.20 | **0.00±0.00** |
| | CatBoost | **0.7650±0.0173** | **0.6750±0.0167** | **0.6888±0.0196** | **0.4951±0.0382** | **0.12±0.02** | **0.00±0.00** |
| | OQBoost | 0.7550±0.0350 | 0.6583±0.0309 | 0.6715±0.0368 | 0.5042±0.0405 | 0.42±0.12 | **0.00±0.00** |
| **CoverType** | XGBoost | 0.9704±0.0008 | 0.9392±0.0042 | 0.9460±0.0034 | 0.0789±0.0017 | **77.56±1.64** | 4.32±0.14 |
| | LightGBM | 0.9704±0.0008 | 0.9397±0.0052 | 0.9466±0.0045 | 0.0823±0.0021 | 284.65±3.53 | 37.64±1.55 |
| | CatBoost | 0.9588±0.0005 | 0.9303±0.0038 | 0.9371±0.0038 | 0.1171±0.0017 | 138.55±5.64 | **0.22±0.01** |
| | OQBoost | **0.9746±0.0007** | **0.9478±0.0038** | **0.9534±0.0034** | **0.0785±0.0013** | 237.95±3.37 | 3.06±0.11 |
| **Higgs** | XGBoost | 0.7304±0.0052 | 0.7291±0.0054 | 0.7293±0.0053 | 0.5259±0.0045 | **5.78±0.82** | 0.08±0.02 |
| | LightGBM | 0.7319±0.0037 | 0.7307±0.0037 | 0.7309±0.0037 | 0.5255±0.0044 | 31.80±3.99 | 0.47±0.08 |
| | CatBoost | 0.7293±0.0055 | 0.7279±0.0055 | 0.7281±0.0055 | 0.5296±0.0043 | 12.41±2.47 | **0.01±0.00** |
| | OQBoost | **0.7328±0.0023** | **0.7316±0.0026** | **0.7317±0.0025** | **0.5247±0.0051** | 47.61±0.41 | 0.88±0.12 |
| **Rotated Synth.** | XGBoost | 0.9758±0.0014 | 0.9758±0.0015 | 0.9758±0.0014 | 0.0819±0.0023 | **6.40±0.18** | 0.10±0.00 |
| | LightGBM | 0.9763±0.0017 | 0.9763±0.0017 | 0.9763±0.0017 | 0.0835±0.0023 | 17.55±0.61 | 0.27±0.02 |
| | CatBoost | 0.9772±0.0019 | 0.9772±0.0019 | 0.9772±0.0019 | 0.0818±0.0034 | 14.89±1.97 | **0.02±0.00** |
| | OQBoost | **0.9794±0.0014** | **0.9794±0.0014** | **0.9794±0.0014** | **0.0736±0.0037** | 25.14±5.81 | 0.42±0.12 |

### 하이라이트 지표 (Highlights)

- **Higgs 전 지표 석권:** OQBoost는 Higgs에서 모든 지표 1위 — 정확도 **0.7328±0.0023**, 균형 정확도 **0.7316±0.0026**, 그리고 최저 로그 손실 **0.5247±0.0051**(LightGBM 0.5255, XGBoost 0.5259, CatBoost 0.5296 대비)을 기록했습니다.
- **CoverType 데이터셋 1위:** OQBoost는 정확도 **0.9746±0.0007** 및 균형 정확도 **0.9478±0.0038**(LightGBM 0.9397, XGBoost 0.9392 능가)을 달성했으며, 로그 손실 역시 최저 수치인 **0.0785±0.0013**(XGBoost 0.0789, CatBoost 0.1171 대비)을 기록했습니다.
- **Rotated Synthetic 강건성 검증:** 회전 공간에서 오블리크 분기의 진가가 입증되어 베이스라인 전체(CatBoost 0.9772, LightGBM 0.9763, XGBoost 0.9758)를 누르고 **0.9794±0.0014**의 가장 높은 균형 정확도 및 **0.0736±0.0037**의 최저 로그 손실을 기록했습니다.
- **`class_weight="balanced"`로 소수 클래스 재현율 향상:** 사전 확률 보정 의사결정 규칙이 Adult에서 균형 정확도를 **0.8332**, CoverType에서 **0.9533**까지 끌어올리며, 로그 손실 손해는 없습니다(학습 자체는 가중치 미적용 → 확률값 보정 유지).

### 그림 1 — 균형 정확도 비교 (Balanced Accuracy Comparison)

![Balanced Accuracy](https://raw.githubusercontent.com/cree1116/oqboost/main/benchmark/results/figures/fig2_balanced_accuracy.png)

### 그림 2 — 성능 대 학습 속도 비교 (Performance vs Training Cost)

![Perf vs Cost](https://raw.githubusercontent.com/cree1116/oqboost/main/benchmark/results/figures/fig5_perf_vs_cost.png)

### 그림 3 — 회전 강건성 평가 (Rotation Robustness)

![Rotation Robustness](https://raw.githubusercontent.com/cree1116/oqboost/main/benchmark/results/figures/fig6_rotation_robustness.png)

### 그림 4 — 결측치 내성 평가 (Missing Value Robustness)

![Missing Value](https://raw.githubusercontent.com/cree1116/oqboost/main/benchmark/results/figures/fig7_missing_value.png)

### 그림 5 — 범주 카디널리티 내성 평가 (Categorical Cardinality Robustness)

![Categorical](https://raw.githubusercontent.com/cree1116/oqboost/main/benchmark/results/figures/fig8_categorical.png)

---

## 알고리즘 개요 (Algorithm)

OQBoost는 **DGCS (Direct Gradient-Covariance Scan, 직접 경사-공분산 스캔)** 알고리즘을 사용하여 수치 최적화나 무작위 확률 사영, 유전적 알고리즘(변이/상속) 없이 결정론적(Deterministic)으로 최적의 사선 분기 방향을 계산합니다.

### DGCS (Direct Gradient-Covariance Scan)
노드 분기를 수행할 때, 다음의 2차 분기 이득 프록시(split-gain proxy) $J(w)$를 최대화하는 사영 방향 $w$를 찾는 것이 목표입니다:
$$J(w) = \frac{(w^T G)^2}{w^T H w + \lambda \|w\|^2}$$
여기서 $G[d] = \sum_{i} x_{id} g_i$는 그라디언트-피처 간의 공분산(gradient-feature covariance)입니다.
헤시안이 고르게 분포한다는 가정($H \approx \sigma I$) 아래, 수식상의 최대점은 다음과 같은 닫힌 형식(closed-form) 솔루션을 가집니다:
$$w^* \propto G$$
OQBoost는 이 닫힌 형식 해를 직접 계산함으로써 복잡한 선형 시스템을 풀거나 수치 최적화 루프를 돌지 않고도 $O(N \cdot d_{\text{sub}})$ 복잡도만으로 최적의 사선 분기 방향을 바로 도출합니다.

### 후보군 (Candidate Families)
데이터의 희소성(Sparsity) 대응 및 다중 클래스 지원을 위해 DGCS는 다음의 사영 방향 후보들을 스캔하여 가장 우수한 분기를 선택합니다:
- **`g` (전체 공분산, Full Covariance)**: 공분산 벡터를 그대로 사용한 방향 $w = G$ (정규화 적용).
- **`s` (부호 공분산, Sign Covariance)**: 부호 정보만을 남긴 희소 방향 $w = \text{sign}(G)$로 강건하고 일반화된 경계를 캡처합니다.
- **`2` / `4` (희소 공분산, Sparse Covariance)**: 공분산 크기 기준 상위 2개 또는 4개의 피처만 사영에 참여시켜 해석 가능성이 높은 희소 사선 분기를 형성합니다.
- **`c` (클래스별 공분산, Per-class Covariance)**: 다중 클래스 환경에서 각 클래스의 개별 그라디언트를 바탕으로 전체 공분산 및 상위 4개 희소 공분산 벡터를 각각 계산하여 모든 클래스의 분별 성능을 보장합니다.

### 방향 캐시 (Direction Cache)
이전 트리들에서 분기를 획득하여 성능이 증명된 최적 방향들은 전역 링 버퍼인 `dir_cache`에 저장됩니다. 대형 노드($N \ge 512$)들은 로컬 경사-공분산 최적점 외에 이 캐시된 전역 우수 방향들을 추가 후보로 스캔하여 학습 효율을 극대화합니다.

수리적 유도 과정은 [`docs/algorithm.md`](docs/algorithm.md)에서, 상세 연구 및 실험 로그는 [`docs/THEORY.md`](docs/THEORY.md)에서 확인하실 수 있습니다.

---

## 성능 및 메모리 최적화 (Performance & Memory Optimizations)

OQBoost는 대규모 정형 데이터셋의 고속 학습 및 추론을 지원하도록 백엔드와 래퍼 계층에 대대적인 저수준 최적화가 적용되었습니다.

* **메모리 사전 할당 구조 (Zero-Allocation Oblique Search)**: 사선 방향 벡터 및 임시 작업 버퍼(`dirs_buf`, `samp_e_buf`, `scratch_cg_s` 등)를 컨텍스트(`OQBoostCtx`) 생성 시점에 미리 할당합니다. 트리가 성장하는 루프 안에서 발생하는 잦은 힙(Heap) 메모리 동적 할당/해제 오버헤드를 근본적으로 제거했습니다.
* **스택 메모리 기반 다중 클래스 연산**: 스택에 할당된 컴팩트 버퍼(`thread_Gc_stack` 등, 최대 64개 클래스)를 사용하여 다중 클래스의 그라디언트와 NaN-routing 통계를 수집합니다. 이를 통해 멀티스레드 OpenMP 축 스캔 시 힙 메모리 접근 및 경합을 피합니다.
* **이진 클래스 고속 패스 및 Softmax 캐싱**: 이진 분류(`K=2`)를 위한 전용 고속 연산 패스를 갖추어 이중 지수 함수(`std::exp`) 연산을 방지합니다. 다중 클래스 연산 시 지수 함수 계산 결과를 스택 버퍼(`p_buf[64]`)에 캐싱하여 고비용 초월 함수 연산을 최소화했습니다.
* **순수 수치형 데이터 제로-카피 추론**: 범주형 변수가 없는 순수 수치형 데이터셋 예측 시(`D_num == D`), 임시 데이터 복사 과정 없이 입력 메모리 버퍼를 그대로 사용하여 불필요한 동적 메모리 할당 및 복사를 생략합니다.
* **Python 계층 내 In-place 연산 적용**: 그라디언트/헤시안 연산 및 Huber 손실 계산 시 임시 넘파이 배열을 생성하지 않고 사전 할당된 버퍼에 복사 및 덮어쓰기 연산(`out=`, `np.clip` 등)을 적용해 메모리 할당 오버헤드를 억제했습니다.
* **범주형 피처 캐시**: 범주형 피처의 인덱스 확인 연산을 Python 클래스 수준에서 캐싱(`_cat_idx_cache_`)하여 라운드마다 반복되는 $O(D)$ 루프 검색을 생략했습니다.
* **실시간 메모리 회수**: 노드 분기가 끝나는 즉시 부모 노드가 점유하던 샘플 인덱스 목록(`node_samp`)을 비우고 메모리를 즉시 반환(`shrink_to_fit`)합니다. 이를 통해 전체 탐색 과정의 피크 메모리 사용량을 $O(N \times \text{current\_depth})$ 수준으로 타이트하게 억제합니다.

---

## 매개변수 명세 (Parameters)

| 매개변수 (Parameter) | 기본값 (Default) | 설명 (Description) |
|-----------|---------|-------------|
| `n_estimators` | 1000 | 부스팅 반복 라운드 수 (트리 개수) |
| `learning_rate` | 0.03 | 트리가 앙상블에 기여하는 학습 축소 계수 |
| `max_depth` | 6 | 리프 노드 수 한도 = $2^{\text{max_depth}}$ (64개) |
| `reg_lambda` | 1.0 | 리프 스코어 가중치에 대한 L2 규제 강도 |
| `reg_alpha` | 0.0 | 리프 가중치에 대한 L1 규제 (그라디언트 소프트 임계) |
| `subsample` | 0.8 | 각 트리 빌드 시 무작위 샘플링 비율 (`goss=True`이면 무시) |
| `goss` | True | Gradient-based One-Side Sampling — 큰 그라디언트 행은 전부 유지, 나머지는 서브샘플링 |
| `goss_top_rate` | 0.2 | GOSS: 항상 유지할 큰 그라디언트 행의 비율 |
| `goss_other_rate` | 0.1 | GOSS: 나머지 행 중 샘플링할 비율 |
| `gamma` | 0.0 | 분기를 수행하기 위한 최소 게인 임계치 |
| `min_child_weight` | 1.0 | 자식 노드의 최소 헤시안 합 |
| `max_leaves` | None | 리프 단위(leaf-wise) 리프 개수 상한 (None = `2^max_depth`) |
| `max_bin` | 255 | 히스토그램 빈(bin) 개수 |
| `colsample_bynode` | 1.0 | 노드별 피처 서브샘플링 비율 |
| `multi_strategy` | "shared" | 다중분류: `"shared"`(라운드당 공유 트리 1개, 빠름) 또는 `"ovr"`(라운드당 K개) |
| `early_stopping_rounds` | 50 | 검증 데이터 손실이 정체될 시 조기 종료 임계치 |
| `cat_features` | None | 범주형 피처 컬럼 명칭(DataFrame) 또는 컬럼 인덱스 목록 |
| `class_weight` | None | `"balanced"` 설정 시 사전 확률 보정 의사결정 규칙 적용 (확률값은 보정 유지) |
| `prior_alpha` | 0.5 | balanced 보정 강도: 0 = 순수 argmax, 1 = 완전 사전확률 보정 |
| `inherited_rp_ratio` | 1.0 | 사용되지 않음 (Legacy, v0.1.8부터 폐지) |
| `mutation_rate` | 0.1 | 사용되지 않음 (Legacy, v0.1.8부터 폐지) |
| `mutation_strength` | 0.5 | 사용되지 않음 (Legacy, v0.1.8부터 폐지) |
| `pobs` | False | 사용되지 않음 (Legacy, v0.1.8부터 폐지) |
| `random_state` | None | 난수 고정 시드 |
| `verbose` | False | 학습 과정 모니터링 출력 여부 |

`OQBoostRegressor`는 위 트리/샘플링 매개변수를 동일하게 받으며, 추가로 `loss`(`"squared_error"` 또는 `"huber"`)와 `huber_delta`(1.0)를 지원합니다.

---

## 벤치마크 실행 방법 (Running Benchmarks)

```bash
cd benchmark

# 전체 벤치마크 가동 (Higgs + CoverType 포함 시 몇 시간 소요될 수 있음)
python run_all.py

# 대용량 데이터셋 제외하고 실행
python run_all.py --skip higgs covertype

# 개별 데이터셋 전용 벤치마크 실행
python adult.py
python rotated_synthetic.py
python missing_value_robustness.py
python categorical_robustness.py

# 벤치마크 완료 후 결과 이미지(차트)들 생성
python generate_figures.py

# 벤치마크 완료 후 결과 테이블 생성 (results/summary.md)
python generate_tables.py
```

일부 대량 데이터셋은 아래 링크에서 수동 다운로드하여 데이터 경로에 위치시켜야 합니다.
- **HIGGS**: https://archive.ics.uci.edu/dataset/280 → `benchmark/data/HIGGS.csv.gz`
- **Give Me Some Credit (Kaggle)**: https://www.kaggle.com/competitions/GiveMeSomeCredit → `benchmark/data/cs-training.csv`

---

## 디렉터리 구조 (Repository Structure)

```
oqboost/
├── src/oqboost/
│   ├── __init__.py
│   ├── _classifier.py      # OQBoostClassifier 래퍼
│   ├── _oqboost.py        # C++ 바인딩 + OQBoostTree, OQBoostContext
│   └── _ext/
│       ├── oqboost.cpp    # 핵심 빌딩 엔진, 정렬 스캔, 직렬화
│       ├── oqboost_types.h # OQTree 노드 구조 선언
│       ├── oqboost_core.h # 핵심 수학 및 공유 상수 라이브러리
│       └── liboqboost.dylib / .so / .dll  (컴파일 바이너리)
├── benchmark/
│   ├── *.py                # 데이터셋별 평가 스크립트
│   ├── _utils.py           # 공통 학습/평가 지표 래퍼
│   ├── generate_figures.py
│   ├── generate_tables.py
│   ├── run_all.py
│   └── results/
│       ├── csv/            # 가공 이전 벤치마크 원본 데이터
│       ├── figures/        # 최종 시각화 차트 이미지
│       └── summary.md
├── docs/
│   ├── algorithm.md        # 사선 수학 및 알고리즘 명세
│   ├── api.md              # 파이썬 전체 API 명세서
│   └── quickstart.md       # 빠른 사용법 가이드
├── tests/
│   └── test_classifier.py
└── pyproject.toml
```

---

## 라이선스 (License)

[MIT](LICENSE) — Copyright (c) 2025 cree1116

---

[English Version (영문 버전)](README.md)
