```python
import pandas as pd
import numpy as np
from typing import Dict


def pandas_1d_to_numpy_dict(df: pd.DataFrame) -> dict:
	"""
	주어진 DataFrame df의 각 컬럼명을 key로 하고, 해당 컬럼을 Numpy array로 변환하여
	dictionary 형태로 반환한다.

	Parameters
	----------
	df : pd.DataFrame
		1차원 형태로 처리할 데이터가 담긴 DataFrame이다.

	Returns
	-------
	dict
		- key: df의 각 컬럼명(str)
		- value: 해당 컬럼을 np.ndarray 형태로 변환한 1차원 array

	DataFrame Format
	----------------
	- 각 컬럼은 변환이 가능한 형태여야 한다. (예: numeric, categorical 등)

	Usage Example
	-------------
	>>> import pandas as pd
	>>> df_example = pd.DataFrame({
	...	 "col1": [1, 2, 3],
	...	 "col2": [4, 5, 6]
	... })
	>>> result = pandas_1d_to_numpy_dict(df_example)
	>>> print(result)
	{'col1': array([1, 2, 3]), 'col2': array([4, 5, 6])}

	"""
	# 복사본 생성(원본 보호)
	df = df.copy()
	return {col: df[col].to_numpy() for col in df.columns}


def pandas_2d_to_numpy_dict(df: pd.DataFrame, id_col: str = "ID") -> Dict[str, np.ndarray]:
	"""
	DataFrame을 받아, 지정된 ID 컬럼(id_col)을 기준으로 groupby 후,
	각 그룹 내 행 순서를 나타내는 cumcount()를 trial_index로 삼아
	나머지 모든 컬럼을 pivot하여 (ID x trial_index) 형태의 2차원 Numpy array로 만든다.

	Parameters
	----------
	df : pd.DataFrame
		2차원 배열 형태로 변환할 원본 DataFrame이다.
	id_col : str, default="ID"
		그룹화에 사용할 식별자 역할의 컬럼명이다.

	Returns
	-------
	dict
		- key: df의 각 컬럼명(str)
		- value: (ID x trial_index) 형태로 pivot된 2차원 np.ndarray

	DataFrame Format
	----------------
	- id_col로 지정된 컬럼을 포함해야 한다.
	- id_col 컬럼 외의 모든 컬럼은 pivot 대상이 된다.

	Usage Example
	-------------
	>>> import pandas as pd
	>>> df_example = pd.DataFrame({
	...	 "ID": [1, 1, 2, 2],
	...	 "colA": [10, 20, 30, 40],
	...	 "colB": [100, 200, 300, 400]
	... })
	>>> arr_dict = pandas_2d_to_numpy_dict(df_example, id_col="ID")
	>>> print(arr_dict["colA"])
	[[10 20]
	 [30 40]]
	>>> print(arr_dict["colB"])
	[[100 200]
	 [300 400]]

	"""
	# 복사본 생성(원본 보호)
	df = df.copy()

	# 그룹별 누적 개수 계산 -> trial_index
	df["trial_index"] = df.groupby(id_col).cumcount()

	# 결과 저장용 dict
	arr_dict = {}

	# ID 컬럼, trial_index 컬럼을 제외한 나머지 컬럼들 pivot
	for col in df.columns:
		if col not in [id_col, "trial_index"]:
			pivoted = df.pivot(index=id_col, columns="trial_index", values=col)
			arr_dict[col] = pivoted.to_numpy()

	return arr_dict
```