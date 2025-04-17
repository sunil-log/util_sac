import numpy as np
import pandas as pd

from util_sac.dict.json_manager import save_json
from util_sac.sys.files.dir_manager import create_dir

"""
see: [[concept, MIL, part xx, design DataLoader (pytorch) for multimodal trials]]
"""

"""
이 모듈은 pandas DataFrame 또는 numpy ndarray 내의 값들을 Python native scalar 혹은 고유한 integer mapping으로 변환하기 위한 유틸리티 함수를 제공한다. 특히 pandas column이 string 자료형인 경우, PyTorch tensor로의 변환을 위해 int로 변환하는 과정을 지원한다.

목적:
  - numpy generic(int64, float64 등)을 Python native scalar로 변환한다.
  - 주어진 데이터의 고유값(unique values)에 대한 integer mapping을 생성한다.
  - 이미 생성된 mapping이 있다면 데이터에 적용한다.

결과:
  - cast_np_scalar(value): numpy generic 객체를 Python native scalar(int, float 등)로 변환한다.
  - create_unique_map(trials): trials 내 고유값에 대한 mapping(dict)을 생성한다.
  - apply_or_create_map(trials, map_name, map_dir, mapping): data를 integer로 매핑한 결과 Series와 mapping을 반환한다. 또한 mapping을 JSON 파일로 저장한다.

예시:

  >>> # 2) create_unique_map
  >>> trials = pd.Series(["apple", "banana", "apple", "cherry"])
  >>> mapping_dict = create_unique_map(trials)
  >>> print(mapping_dict)
  {'apple': 0, 'banana': 1, 'cherry': 2}

  >>> # 3) apply_or_create_map
  >>> mapped_series, used_mapping = apply_or_create_map(trials, map_name='fruit_map')
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
	"""
	numpy의 다양한 scalar 타입(e.g., int64, float64, "U<28" 등)을
	Python의 native scalar로 변환한다. 이는 dict의 key로 활용할 때
	np.generic 그대로 사용하면 hashability 문제 등이 발생할 수 있기 때문이다.
	따라서 .item() 메서드를 통해 numpy scalar를 일반적인 Python int, float, str 등으로
	변환하여 일관성 있는 dict key로 사용할 수 있게 한다.

	Parameters
	----------
	value : Any
	    변환 대상. 주로 numpy scalar 타입.

	Returns
	-------
	scalar
	    Python int, float, str 등 native scalar 타입.
	"""
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


def apply_or_create_map(data, map_name, root_dir='./', mapping=None, return_type='numpy'):
	"""
	trials: numpy array (다차원 가능) 또는 pandas Series
	mapping: 이미 생성된 mapping dict (선택)

	1) data가 numpy array라면:
	   - ndim이 2 이상일 경우 flatten하여 Series로 변환한다.
	   - 변환된 shape 정보를 보관한다.
	2) data가 pandas Series라면 그대로 사용한다.
	3) mapping이 None이라면 create_unique_map을 이용해 생성한다.
	4) 해당 mapping을 trials(혹은 flatten된 trials)에 적용한다.
	5) 다차원 numpy array였다면, 적용된 결과를 다시 원래 shape로 reshape한다.
	6) 최종 결과와 mapping을 반환한다.
	7) 생성 혹은 사용된 mapping은 map_dir/map_name.json으로 저장한다.
	"""
	original_shape = None
	flattened = False

	# numpy array 처리
	if isinstance(data, np.ndarray):
		# 다차원일 경우 flatten
		if data.ndim > 1:
			original_shape = data.shape
			data = data.flatten()
			flattened = True
		data = pd.Series(data)

	# pandas Series 처리
	elif not isinstance(data, pd.Series):
		raise TypeError("입력 데이터는 numpy array 또는 pandas Series만 지원한다.")

	# mapping이 없다면 새로 생성
	if mapping is None:
		mapping = create_unique_map(data)

	# 매핑을 JSON으로 저장
	map_dir = f"{root_dir}/maps"
	create_dir(map_dir)
	save_json(mapping, f"{map_dir}/{map_name}.json")

	# mapping 적용 후 int64로 변환
	mapped_series = data.map(mapping).astype(np.int64)

	# return_type에 따라 반환
	if return_type == 'series':
		return mapped_series, mapping
	else:  # 'numpy'
		if flattened and original_shape is not None:
			mapped_array = mapped_series.values.reshape(original_shape)
		else:
			mapped_array = mapped_series.values
		return mapped_array, mapping