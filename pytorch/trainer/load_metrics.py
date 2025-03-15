
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""


import pandas as pd



def concat_train_test_metrics(base_path, keywords):
	"""
	주어진 keywords 리스트를 모두 포함하는 디렉토리를 찾아, 해당 디렉토리 내부의 train_test_metrics.csv 파일을 읽은 후,
	각 DataFrame에 디렉토리 이름을 'fn' 컬럼으로 추가하고, 모든 DataFrame을 axis=0으로 concat하여 반환합니다.

	Parameters:
	- keywords (list of str): 디렉토리 이름에 포함되어야 하는 문자열들의 리스트

	Returns:
	- pd.DataFrame: 모든 csv 데이터를 합친 DataFrame
	"""
	dfs = []
	# 현재 디렉토리 내의 모든 폴더 순회
	for directory in base_path.iterdir():
		if directory.is_dir() and all(kw in directory.name for kw in keywords):
			csv_path = directory / 'train_test_metrics.csv'
			if csv_path.exists():
				df = pd.read_csv(csv_path)
				df['fn'] = directory.name
				dfs.append(df)
	if dfs:
		return pd.concat(dfs, axis=0, ignore_index=True)
	else:
		return pd.DataFrame()



