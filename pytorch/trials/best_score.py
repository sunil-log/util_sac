
# -*- coding: utf-8 -*-
"""
Created on  Apr 17 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.sys.date_format import add_timestamp_to_string
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.print_array import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette


from scipy.stats import zscore



def smooth_series(s: pd.Series, method: str = "ema", **kw) -> pd.Series:
	"""
	시계열을 평활(smoothing)하여 잡음을 줄이고 추세를 파악하도록 하는 유틸리티 함수이다.

	Parameters
	----------
	s : pd.Series
		입력 시계열. 인덱스는 그대로 보존된다.
	method : {"ema", "sma"}, default "ema"
		적용할 평활 방식
		* **"ema"** – Exponential Moving Average. 지수 가중치를 사용한다.
		* **"sma"** – Simple Moving Average. 고정 window 내 단순 평균을 취한다.
	**kw : dict, optional
		추가 매개변수. `method`에 따라 다음 key를 지원한다.
		* **"ema"**
		  - `alpha` (float, default 0.3) : 지수가중치 계수(0 < alpha ≤ 1).
		* **"sma"**
		  - `w` (int, default 5) : window 길이(양의 정수). `min_periods`는 1로 고정되어 있어 초반 구간에서도
			값이 계산된다.

	Returns
	-------
	pd.Series
		입력과 동일한 길이를 갖는 평활 시계열.

	Notes
	-----
	* **길이는 변하지 않는다.** `min_periods=1` 설정으로 인해 초반 (window − 1) 구간에서도 평균을 계산하며,
	  이때는 사용 가능한 샘플 수(< window)가 그대로 활용된다. 따라서 초기 값들은
	  사실상 expanding mean에 가까워 **통계적 안정성이 상대적으로 낮다**는 점을 유의한다.
	* EMA는 `pandas.Series.ewm`, SMA는 `pandas.Series.rolling`을 내부적으로 호출한다.

	Examples
	--------
	>>> # EMA smoothing
	>>> smoothed = smooth_series(price, method="ema", alpha=0.2)
	>>> # SMA smoothing
	>>> smoothed = smooth_series(signal, method="sma", w=7)
	"""
	if method == "ema":
		return s.ewm(alpha=kw.get("alpha", 0.3)).mean()
	if method == "sma":
		return s.rolling(window=kw.get("w", 5), min_periods=1).mean()
	raise ValueError(f"지원하지 않는 method: {method}")




def compute_weighted_score(
	df: pd.DataFrame,
	metrics_cfg: dict[str, dict],
	*,
	normalize: str = "z-score",
	add_col: str | None = None
) -> tuple[pd.Series, int]:
	"""
	목적
	----
	여러 지표(metric)의 **평활(smoothing)·전처리·가중합**을 자동으로 수행하여
	각 epoch별 종합 점수를 산출하고, **가장 우수한 모델(epoch)**을 선택한다.
	‒ loss·AUC·F1 등 방향이 다른 지표를 한데 묶어 비교 가능하도록
	  *로그 변환*, *방향 반전*, *정규화*를 지원한다.
	‒ 지표마다 개별 **smoothing**(EMA/SMA) 파라미터를 지정할 수 있어
	  노이즈를 줄인 후 안정적인 선택 기준을 제공한다.

	Parameters
	----------
	df : pd.DataFrame
		지표들이 열(column)로 정리된 학습 기록.
	metrics_cfg : dict[str, dict]
		지표별 설정. key = 열 이름, value = 아래 항목 포함 dict.

		* **weight** (float)              : 가중치.
		* **direction** {"max","min"}    : 'min'이면 값에 -1 곱해 방향 통일.
		* **log** (bool)                 : True → log1p 변환.
		* **norm** {"inherit","none","z-score","minmax"} (선택)
		  - "inherit" → 함수 인수 `normalize`와 동일 방식 적용.
		* **smooth** (선택)
		  ```python
		  "smooth": {
		  	"method": {"ema","sma"},
		  	"kw": {...}       # smooth_series 에 넘길 추가 파라미터
		  }
		  ```

	normalize : {"z-score","minmax","none"}, default "z-score"
		전역 스케일 정규화 방법.
	add_col : str | None, default None
		None이 아니면 df[add_col]에 계산된 종합 점수를 저장한다.

	Returns
	-------
	score_series : pd.Series
		epoch별 가중합 점수.
	best_idx : int
		점수가 최대인 epoch(index).

	예시
	----
	>>> metric_cfg = {
	...     "valid_loss": {
	...         "weight": 0.4, "direction": "min", "log": True,
	...         "smooth": {"method": "ema", "kw": {"alpha": 0.2}}
	...     },
	...     "auc_roc_valid": {
	...         "weight": 0.35, "direction": "max",
	...         "smooth": {"method": "sma", "kw": {"w": 3}}
	...     },
	...     "f1_class_macro_valid": {
	...         "weight": 0.25, "direction": "max"
	...     }
	... }
	>>> score, best_epoch = compute_weighted_score(
	...     df, metric_cfg, normalize="z-score", add_col="score"
	... )
	>>> print(f"Best epoch = {best_epoch}, score = {score[best_epoch]:.4f}")

	Notes
	-----
	* smoothing → log → 방향 통일 → 정규화 → 가중합 순서로 처리된다.
	* σ=0 또는 range=0인 지표는 0으로 대체하여 정규화 오류를 방지한다.
	"""
	def _normalize(s: pd.Series, how: str):
		if how == "none":
			return s
		if how == "minmax":
			rng = s.max() - s.min()
			return (s - s.min()) / rng if rng != 0 else 0.0
		std = s.std(ddof=0)
		return (s - s.mean()) / std if std != 0 else 0.0

	parts = []

	for col, cfg in metrics_cfg.items():
		if col not in df.columns:
			raise KeyError(f"'{col}' not found in df columns.")
		s = df[col].copy()

		# --- 1. smoothing ---------------------------------------------------
		if "smooth" in cfg:
			sm_opt = cfg["smooth"]
			s = smooth_series(
				s,
				method=sm_opt.get("method", "ema"),
				**sm_opt.get("kw", {})
			)

		# --- 2. log 변환 -----------------------------------------------------
		if cfg.get("log", False):
			s = np.log1p(s.clip(lower=0))

		# --- 3. 방향 통일 ----------------------------------------------------
		if cfg.get("direction", "max") == "min":
			s = -s

		# --- 4. 정규화 -------------------------------------------------------
		local_norm = cfg.get("norm", "inherit")
		s = _normalize(s, normalize if local_norm == "inherit" else local_norm)

		# --- 5. 가중치 -------------------------------------------------------
		parts.append(cfg.get("weight", 1.0) * s)

	score_series = sum(parts)
	if add_col is not None:
		df[add_col] = score_series
	best_idx = int(score_series.idxmax())
	return score_series, best_idx