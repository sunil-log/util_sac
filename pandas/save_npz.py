


import numpy as np
import pandas as pd


def save_df_columns_as_npz(df, filename):
	"""
	주어진 데이터프레임(df)의 모든 컬럼을 NumPy 배열로 변환하고 이를 .npz 파일로 저장합니다.

	Parameters:
	df : pandas.DataFrame
		변환하고 저장할 데이터프레임.
	filename : str
		저장할 파일의 이름 (확장자 .npz 포함).
	"""
	# 데이터프레임의 각 컬럼을 NumPy 배열로 변환
	arrays = {col: np.array(df[col].tolist()) for col in df.columns}

	# 변환된 배열을 .npz 파일로 저장
	np.savez(filename, **arrays)


def print_dict_contents(d):

	print(f"========== Data Loaded ============")
	for key in list(d.keys()):
		print(f"> {key}: {d[key].shape}", end=", ")
		if len(d[key].shape) == 1:
			print(d[key][:3], end="...")
			print(d[key][-3:], end="")
		print()


def load_all_arrays_from_npz(filename):
	"""
	.npz 파일에서 모든 배열을 로드합니다.

	Parameters:
	filename : str
		로드할 .npz 파일의 경로.

	Returns:
	dict
		파일 내 모든 배열을 포함하는 사전. 각 키는 .npz 파일 내의 배열 이름입니다.
	"""
	# .npz 파일 로드
	data = np.load(filename)

	# 파일 내의 모든 배열을 사전 형태로 반환
	d = {file: data[file] for file in data.files}

	# print
	print(f"> {filename}")
	print_dict_contents(d)

	return d


def load_df_from_npz(fn):
	"""
	.npz 파일로부터 로드된 데이터 사전을 pandas DataFrame으로 변환합니다.

	Parameters:
	data_dict : dict
		.npz 파일로부터 로드된 데이터가 포함된 사전.

	Returns:
	pandas.DataFrame
		변환된 데이터프레임.

	주의: 모든 array 가 1차원일 때만 제대로 작동합니다.
	"""
	# 사전의 각 배열을 DataFrame의 컬럼으로 변환
	data_dict = load_all_arrays_from_npz(fn)
	return pd.DataFrame(data_dict)

def concat_dicts(*dicts):
	"""
	여러 사전의 배열을 axis=0을 기준으로 연결하여 새로운 사전을 생성합니다.
	모든 사전은 같은 키를 가지고 있으며, 각 키에 해당하는 값은 NumPy 배열입니다.

	Parameters:
	dicts : dict
		연결할 사전들.

	Returns:
	dict
		연결된 배열을 포함하는 새로운 사전.
	"""
	# 새로운 사전 초기화
	concatenated_dict = {}

	# 모든 키에 대해 반복
	for key in dicts[0]:
		# 같은 키의 배열들을 axis=0을 기준으로 연결
		concatenated_dict[key] = np.concatenate([d[key] for d in dicts], axis=0)

	return concatenated_dict

def load_multiple_npz(fns):

	ds = []
	for fn in fns:
		d = load_all_arrays_from_npz(fn)
		ds += [d]

	# concat axis=0
	d = concat_dicts(*ds)

	print(f"> {fns}")
	print_dict_contents(d)

	return d