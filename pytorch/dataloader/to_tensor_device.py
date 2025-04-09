import numpy as np
import torch


def dict_to_tensors(d, float_dtype=torch.float32, int_dtype=torch.int64):
	"""
	주어진 dictionary의 value들을 torch Tensor로 변환한다.
	float, int 모두 기본적으로 float_dtype, int_dtype 로 변환한다.

	Args:
		d (dict): 여러 형태(tensor, numpy array, list, 숫자 등)의 데이터를 value로 가지는 dictionary.
		float_dtype (torch.dtype): float 형 변환 시 사용할 dtype (기본값: torch.float32).
		int_dtype (torch.dtype): int 형 변환 시 사용할 dtype (기본값: torch.int64).

	Returns:
		dict: value들이 모두 torch Tensor로 변환된 dictionary.
	"""
	new_d = {}
	for k, v in d.items():
		if isinstance(v, torch.Tensor):
			# 이미 torch Tensor인 경우
			if torch.is_floating_point(v):
				new_d[k] = v.to(dtype=float_dtype)
			else:
				new_d[k] = v.to(dtype=int_dtype)
		elif isinstance(v, np.ndarray):
			# numpy array인 경우
			if np.issubdtype(v.dtype, np.floating):
				new_d[k] = torch.from_numpy(v).to(dtype=float_dtype)
			else:
				new_d[k] = torch.from_numpy(v).to(dtype=int_dtype)
		elif isinstance(v, list):
			# list인 경우, 원소가 부동소수점인지 정수인지 판단
			if all(isinstance(x, float) for x in v):
				new_d[k] = torch.tensor(v, dtype=float_dtype)
			else:
				new_d[k] = torch.tensor(v, dtype=int_dtype)
		elif isinstance(v, (int, float, np.number)):
			# 숫자인 경우, 1개의 element를 가지는 tensor로 변환
			if isinstance(v, (float, np.floating)):
				new_d[k] = torch.tensor([v], dtype=float_dtype)
			else:
				new_d[k] = torch.tensor([v], dtype=int_dtype)
		else:
			raise TypeError(f"Unsupported data type for key '{k}': {type(v)}")

	return new_d


def move_dict_tensors_to_device(d, device):
	"""
	dictionary 내 value들이 이미 torch Tensor라고 가정하고,
	모두 지정된 device로 옮긴다.

	Args:
		d (dict): value가 torch Tensor인 dictionary.
		device (torch.device): 데이터를 옮길 대상 device.

	Returns:
		dict: value들이 지정된 device로 옮겨진 dictionary.
	"""
	new_d = {}
	for k, v in d.items():
		if not isinstance(v, torch.Tensor):
			raise TypeError(f"Value for key '{k}' is not a torch.Tensor. Found: {type(v)}")

		new_d[k] = v.to(device)
	return new_d
