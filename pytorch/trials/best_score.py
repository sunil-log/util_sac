
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




def calc_best_auc(df):


	print(df)
	exit()


	"""
	일반적으로 모이는 지표들
	epoch,
		train_loss, valid_loss, test_loss,
		f1_class_0_train, f1_class_1_train, f1_class_macro_train,
		f1_class_0_valid, f1_class_1_valid, f1_class_macro_valid,
		f1_class_0_test, f1_class_1_test, f1_class_macro_test,
		auc_roc_train, auc_roc_valid, auc_roc_test, lr

	1. auc_roc_valid (높을수록 좋음)
	2. valid_loss (낮을수록 좋음)
	3. gap_auc: auc_roc_valid - auc_roc_train (낮을수록 좋음)
	4. auc 안정성:

	실제로 사용하는 지표: auc_roc_valid, valid_loss, auc_roc_train
	"""

	# load df
	metrics = ["auc_roc_valid", "valid_loss", "auc_roc_train", "auc_roc_test", "f1_class_0_valid"]
	df = df[metrics].copy()

	"""
	Smooth the series
	"""
	metrics_s = []
	for m in metrics:
		m_new = f"{m}_s"
		df[m_new] = smooth_series(df[m], method="ema", alpha=0.2)
		metrics_s.append(m_new)




	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, len(metrics), figsize=(len(metrics)*5, 6))
	for i, m in enumerate(metrics):
		ax[i].plot(df[m], label=m)
		ax[i].plot(df[f"{m}_s"], label=f"{m}_s")
		ax[i].set_title(m)
		ax[i].set_xlabel("epoch")
		ax[i].set_ylabel(m)
		ax[i].grid()
		ax[i].legend()





	plt.savefig("smoothing.png", bbox_inches='tight')
	exit()












def main():
	pass

if __name__ == "__main__":
	main()
