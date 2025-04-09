import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def t_test_for_binary_features(X, y, feature_names, correction='fdr_bh'):
	"""
	이 함수는 y가 binary class일 때 각 feature에 대해 t-test를 수행하고,
	지정된 다중비교(multi-comparison) 보정 방법에 따라 p-value를 보정합니다.

	Parameters
	----------
	X : NumPy Array
		Feature matrix
	y : NumPy Array
		Binary class (0 또는 1)
	feature_names : list
		Feature 이름 리스트
	correction : str, optional
		다중비교 보정 방법 (기본값: 'none')
		지원하는 값:
			- 'none'	  : 보정하지 않음
			- 'bonferroni': Bonferroni 보정
			- 'fdr_bh'	: False Discovery Rate 보정 (Benjamini/Hochberg)
			- 'holm'	  : Holm-Bonferroni 보정

	Returns
	-------
	df_results : pandas DataFrame
		각 feature마다 t-statistic, raw p-value, (보정된 p-value)를 담은 DataFrame
	"""
	results = []
	p_values = []

	for i, feature in enumerate(feature_names):
		group0 = X[y == 0, i]
		group1 = X[y == 1, i]
		t_stat, p_val = ttest_ind(group0, group1)
		results.append((feature, t_stat, p_val))
		p_values.append(p_val)

	# 다중비교 보정 적용
	if correction == 'none':
		p_values_corrected = p_values
	else:
		_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method=correction)

	df_results = pd.DataFrame({
		"feature": [r[0] for r in results],
		"t-statistic": [r[1] for r in results],
		"p-value": [r[2] for r in results],
		"corrected p-value": p_values_corrected
	})

	return df_results


if __name__ == '__main__':

	df_feature = pd.DataFrame(features)
	X = df_feature.values
	y = np.array(y)
	feature_names = df_feature.columns
	"""
	X          NumPy Array          (185, 15)                   21.68 KB float64
	y          NumPy Array          (185,)                       1.45 KB int64
	"""

	df_res = t_test_for_binary_features(X, y, feature_names, correction='fdr_bh')
	df_res = df_res.sort_values(by='corrected p-value')
	print_partial_markdown(df_res)
