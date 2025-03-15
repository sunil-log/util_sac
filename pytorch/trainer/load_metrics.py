
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac


from util_sac.pytorch.trainer.load_metrics import load_metrics

# read all train_test_metrics.csv files
base_path = Path('./trials')  # 현재 작업 디렉토리
keywords = ['2025-03-15', __ID_2151__]
result_df = load_metrics(base_path, keywords)
print(result_df)

"""


import pandas as pd


def parse_fn(fn_str: str) -> dict:
	"""
	__로 분할:
	  - 첫 번째 segment는 timestamp로 저장한다.
	  - 이후 segment 각각에 대해:
		- '_'가 없으면 무시
		- '_'가 하나 이상이면, 오른쪽에서 한 번만 split하여
		  앞부분은 key, 뒷부분은 value로 저장한다.
	"""
	parts = fn_str.split("__")
	result = {}

	# 맨 앞 segment는 날짜/시간 정보라 가정하고 "timestamp"로 저장
	result["timestamp"] = parts[0]

	# 나머지 segment들을 순회하며 key-value 추출
	for part in parts[1:]:
		if "_" not in part:
			# '_'가 없으면 무시
			continue
		key, val = part.rsplit("_", 1)  # 오른쪽에서 한 번만 split
		result[key] = val
		print(result)
		exit()

	return result


def load_metrics(base_path, keywords, parse=True):
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

	if len(dfs) < 1:
		print(f"No directory with keywords {keywords} found in {base_path}")
		return None

	df = pd.concat(dfs, axis=0, ignore_index=True)

	if parse:
		parsed_df = df["fn"].apply(parse_fn).apply(pd.Series)
		df = pd.concat([df, parsed_df], axis=1)

	return df






