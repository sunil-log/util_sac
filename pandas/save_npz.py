import numpy as np
import pandas as pd

def save_df_as_npz(df, filename):
	"""
	주어진 DataFrame(df)의 모든 컬럼을 NumPy 배열로 변환하고, 이를 .npz 파일로 저장한다.

	Parameters
	----------
	df : pandas.DataFrame
		저장할 DataFrame
	filename : str
		저장할 파일 이름(확장자 .npz 포함)
	"""
	# 각 컬럼을 배열로 변환
	arrays = {col: np.array(df[col].tolist()) for col in df.columns}
	# .npz로 저장
	np.savez(filename, **arrays)


def load_df_from_npz(filename):
	"""
	.npz 파일을 로드해 pandas DataFrame으로 변환한다.
	(모든 배열이 1차원이라고 가정)

	Parameters
	----------
	filename : str
		로드할 .npz 파일 경로

	Returns
	-------
	pandas.DataFrame
		변환된 DataFrame
	"""
	data = np.load(filename)
	# 파일 내 모든 배열을 사전(dict)으로 획득
	arrays = {key: data[key] for key in data.files}
	# dict -> DataFrame 변환
	return pd.DataFrame(arrays)


# ----------------- 사용 예시 -----------------
if __name__ == "__main__":
	# 1) 예시 DataFrame 생성
	df = pd.DataFrame({
		'Feature1': [10, 20, 30],
		'Feature2': [0.1, 0.2, 0.3],
		'Label': ['cat', 'dog', 'bird']
	})

	# 2) 저장
	save_df_as_npz(df, "example.npz")

	# 3) 로드 후 DataFrame 확인
	df = load_df_from_npz("example.npz")
	print(df)
