import numpy as np
import pandas as pd
from util_sac.dict.json_manager import save_json


def cast_np_scalar(value):
	# numpy int64, float64 등은 .item()을 통해 파이썬 native scalar로 변환한다.
	if isinstance(value, np.generic):
		return value.item()
	return value


def create_unique_map(data):
	if not isinstance(data, (np.ndarray, pd.Series)):
		raise TypeError("입력 데이터는 numpy array 또는 pandas Series만 지원한다.")

	# numpy array인 경우 np.unique, pandas Series인 경우 Series.unique() 사용
	if isinstance(data, np.ndarray):
		unique_values = np.unique(data)
	else:
		unique_values = data.unique()

	mapping_dict = {cast_np_scalar(val): idx for idx, val in enumerate(unique_values)}
	return mapping_dict


def apply_or_create_map(df, column, mapping=None):
	"""
	df의 특정 column에 대해 mapping이 주어지지 않았을 경우
	create_unique_map을 사용해 새로운 mapping을 생성한 뒤 적용한다.
	"""
	if mapping is None:
		mapping = create_unique_map(df[column])

	# 만들어진 또는 전달받은 mapping을 컬럼에 적용한다.
	df[column] = df[column].map(mapping)

	# 변환 완료 후 df와 사용된 mapping을 반환한다.
	return df, mapping
