import numpy as np
import pandas as pd

from util_sac.dict.json_manager import save_json

"""
see: [[concept, MIL, part xx, design DataLoader (pytorch) for multimodal data]]
"""

"""
이 모듈은 pandas DataFrame 또는 numpy ndarray 내의 값들을 Python native scalar 혹은 고유한 integer mapping으로 변환하기 위한 유틸리티 함수를 제공한다. 특히 pandas column이 string 자료형인 경우, PyTorch tensor로의 변환을 위해 int로 변환하는 과정을 지원한다.

목적:
  - numpy generic(int64, float64 등)을 Python native scalar로 변환한다.
  - 주어진 데이터의 고유값(unique values)에 대한 integer mapping을 생성한다.
  - 이미 생성된 mapping이 있다면 데이터에 적용한다.

결과:
  - cast_np_scalar(value): numpy generic 객체를 Python native scalar(int, float 등)로 변환한다.
  - create_unique_map(data): data 내 고유값에 대한 mapping(dict)을 생성한다.
  - apply_or_create_map(data, map_name, map_dir, mapping): data를 integer로 매핑한 결과 Series와 mapping을 반환한다. 또한 mapping을 JSON 파일로 저장한다.

예시:

  >>> # 2) create_unique_map
  >>> data = pd.Series(["apple", "banana", "apple", "cherry"])
  >>> mapping_dict = create_unique_map(data)
  >>> print(mapping_dict)
  {'apple': 0, 'banana': 1, 'cherry': 2}

  >>> # 3) apply_or_create_map
  >>> mapped_series, used_mapping = apply_or_create_map(data, map_name='fruit_map')
  >>> print(mapped_series)
  0    0
  1    1
  2    0
  3    2
  dtype: int64
  >>> print(used_mapping)
  {'apple': 0, 'banana': 1, 'cherry': 2}
"""



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


def apply_or_create_map(data, map_name, map_dir='./maps', mapping=None):
	"""
	data: numpy array 또는 pandas Series
	mapping: 이미 생성된 mapping dict (option)

	1) data가 numpy array라면 pandas Series로 변환한다.
	2) mapping이 None이라면 create_unique_map을 이용해 생성한다.
	3) 해당 mapping을 data에 적용한 결과 Series와 mapping을 반환한다.
	"""
	# numpy array라면 Series로 변환
	if isinstance(data, np.ndarray):
		data = pd.Series(data)
	elif not isinstance(data, pd.Series):
		raise TypeError("입력 데이터는 np.ndarray 또는 pd.Series만 지원한다.")

	# mapping이 없다면 새로 생성
	if mapping is None:
		mapping = create_unique_map(data)

	# map 을 json 으로 저장
	save_json(mapping, f"{map_dir}/{map_name}.json")

	# mapping을 적용해 변환된 Series 생성
	mapped_series = data.map(mapping)

	# 변환된 시리즈와 적용된 mapping을 반환
	return mapped_series
