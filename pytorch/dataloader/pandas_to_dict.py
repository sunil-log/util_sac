import pandas as pd
import numpy as np
from typing import Dict

"""
이 모듈은 pandas DataFrame을 편리하게 Numpy array로 변환하기 위한 함수를 제공한다.

다음 함수를 포함한다:
- pandas_1d_to_numpy_dict: 입력된 DataFrame의 각 컬럼을 key로 하고, 1차원 형태의 Numpy array를 value로 하는 dict를 반환한다.
- pandas_2d_to_numpy_dict: ID 및 trial_index 기준으로 그룹화한 뒤, (ID x trial_index) 형태의 2차원 Numpy array를 value로 하는 dict를 생성한다.

사용 목적
---------
- pandas DataFrame에서 PyTorch 등 Numerical Computing 환경으로 데이터를 이관할 때 사용된다.
- 기존 DataFrame을 유지한 채로 Numpy array 형태만 별도로 얻어, 이후 수치 연산 및 모델 학습에 활용하기 위함이다.

사용 예시
---------
1. Preprocessing을 pandas로 수행한 뒤, PyTorch 모델 학습을 위해 Numpy array 형태로 변환할 때.
2. Hierarchical 데이터를 (ID x trial_index) 구조로 유지하면서, 2차원 Numpy 형태로 가공해 배치 처리를 용이하게 하고자 할 때.
"""

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

import pandas as pd
import numpy as np
from typing import Dict


def pandas_2d_to_numpy_dict(
	df: pd.DataFrame,
	id_col: str = "ID",
) -> Dict[str, np.ndarray]:

