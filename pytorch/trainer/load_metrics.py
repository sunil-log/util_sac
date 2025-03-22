
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

from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pandas.save_npz import load_df_from_npz
from util_sac.dict.save_args import load_args

from util_sac.sys.search_files import search_items_df



import pandas as pd


def load_hyperparams(base_path):
	"""
	주어진 keywords 리스트를 모두 포함하는 디렉토리를 찾아,
	hyperparameters.json 파일만 읽어서 하나의 DataFrame으로 합친다.
	각 DataFrame에 디렉토리 이름에서 추출한 trial_id를 추가한다.

	Parameters:
	- base_path (pathlib.Path): 탐색할 기준 경로
	- keywords (list of str): 디렉토리 이름에 포함되어야 하는 문자열들의 리스트

	Returns:
	- pd.DataFrame 혹은 None: 모든 json 데이터를 합친 DataFrame.
	  조건에 맞는 디렉토리가 없으면 None을 반환한다.
	"""
	search_pattern = r".*hyperparameters\.json$"  # 정규표현식
	result_df = search_items_df(base_path, search_pattern, search_type="files")
	"""
	|     | Path                                                                                                              | Parent                                                                                       | Name                 | Type   | Extension   |
	|----:|:------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------------------|:-------|:------------|
	|   0 | trials/ID_124024__study_name/2025-03-22_12-42-36__ID_124024__study_name__Trial_29__session_2/hyperparameters.json | trials/ID_124024__study_name/2025-03-22_12-42-36__ID_124024__study_name__Trial_29__session_2 | hyperparameters.json | File   | .json       |
	|   1 | trials/ID_124024__study_name/2025-03-22_12-46-17__ID_124024__study_name__Trial_80__session_1/hyperparameters.json | trials/ID_124024__study_name/2025-03-22_12-46-17__ID_124024__study_name__Trial_80__session_1 | hyperparameters.json | File   | .json       |
	|   2 | trials/ID_124024__study_name/2025-03-22_12-45-14__ID_124024__study_name__Trial_64__session_0/hyperparameters.json | trials/ID_124024__study_name/2025-03-22_12-45-14__ID_124024__study_name__Trial_64__session_0 | hyperparameters.json | File   | .json       |
	"""
	# print_partial_markdown(result_df)


	list_args = []
	for i, row in result_df.iterrows():
		json_path = row['Path']
		args = load_args(json_path)
		list_args.append(args)


	if not list_args:
		print(f"No 'hyperparameters.json' found in {base_path}")
		return None

	# DataFrame 생성
	df = pd.DataFrame(list_args)
	return df



def load_metrics(base_path, keywords):
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
			npz_path = directory / 'train_test_metrics.npz'
			json_path = directory / 'hyperparameters.json'
			if npz_path.exists():
				# load npz
				df = load_df_from_npz(npz_path)
				# load args and assign to df
				args = load_args(json_path)
				df = df.assign(**args)
				# append date
				df['trial_id'] = directory.name.split('__')[0]
				dfs.append(df)

	if len(dfs) < 1:
		print(f"No directory with keywords {keywords} found in {base_path}")
		return None

	df = pd.concat(dfs, axis=0, ignore_index=True)

	return df


def main():

	from pathlib import Path
	from util_sac.pytorch.trainer.load_metrics import load_metrics
	from util_sac.pandas.print_df import print_partial_markdown

	# read all train_test_metrics.csv files
	base_path = Path('./trials')  # 현재 작업 디렉토리
	keywords = ['__ID_2338__']
	df = load_metrics(base_path, keywords)
	"""
	print_partial_markdown(df)
	|      |   epoch |   train_loss |   test_loss |   f1_class_0_train |   f1_class_1_train |   f1_class_macro_train |   f1_class_0_test |   f1_class_1_test |   f1_class_macro_test |     lr |   input_dim |   n_head |   q_dim | trial_id            |
	|-----:|--------:|-------------:|------------:|-------------------:|-------------------:|-----------------------:|------------------:|------------------:|----------------------:|-------:|------------:|---------:|--------:|:--------------------|
	|    0 |       0 |     0.660726 |    0.667487 |          0         |          0.816568  |               0.408284 |         0         |         0.780488  |              0.390244 | 0.001  |           2 |        8 |       2 | 2025-03-15_14-40-08 |
	|    1 |       1 |     0.645456 |    0.66232  |          0         |          0.816568  |               0.408284 |         0         |         0.780488  |              0.390244 | 0.001  |           2 |        8 |       2 | 2025-03-15_14-40-08 |
	|    2 |       2 |     0.631349 |    0.660944 |          0         |          0.816568  |               0.408284 |         0         |         0.780488  |              0.390244 | 0.001  |           2 |        8 |       2 | 2025-03-15_14-40-08 |
	|    3 |       3 |     0.675514 |    0.662108 |          0         |          0.816568  |               0.408284 |         0         |         0.780488  |              0.390244 | 0.001  |           2 |        8 |       2 | 2025-03-15_14-40-08 |
	"""

	# select the best f1_class_macro_test
	df_max = df.loc[df.groupby("trial_id")["f1_class_macro_test"].idxmax()].reset_index(drop=True)
	print_partial_markdown(df_max)



