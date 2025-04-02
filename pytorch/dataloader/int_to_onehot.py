

import numpy as np
import torch
import torch.nn.functional as F

def one_hot_encode(input_data):
	"""
	입력:
		- NumPy Array 또는 PyTorch Tensor (shape 예시: (95, 5, 40))
	처리:
		1. NumPy Array라면 PyTorch Tensor로 변환 (dtype=torch.long)
		2. 이미 PyTorch Tensor라면 그대로 사용하되, long 타입으로 변환
		3. 내부 정수들의 최대값 max_val 구하기
		4. one_hot으로 변환
		   - 최종 shape: (원본 shape) + (max_val + 1)
	반환:
		- one_hot 결과 (PyTorch Tensor)
	"""
	# 1. NumPy Array -> PyTorch Tensor 변환
	if isinstance(input_data, np.ndarray):
		tensor_data = torch.tensor(input_data, dtype=torch.long)
	# 2. PyTorch Tensor 처리
	elif isinstance(input_data, torch.Tensor):
		tensor_data = input_data.long()
	else:
		raise TypeError("input_data는 NumPy Array나 PyTorch Tensor여야 한다.")

	# 3. 내부 정수들의 최대값 구하기
	max_val = torch.max(tensor_data).item()
	num_classes = max_val + 1

	# 4. one-hot 변환
	one_hot_tensor = F.one_hot(tensor_data, num_classes=num_classes)

	return one_hot_tensor



def one_hot_to_int(one_hot_tensor: torch.Tensor) -> torch.Tensor:
	"""
	(batch, feature) 형태의 one-hot Tensor를 해당하는 int로 변환하는 함수.

	Args:
		one_hot_tensor (torch.Tensor): (batch, feature) 형태의 one-hot Tensor

	Returns:
		torch.Tensor: 각 row마다 one-hot에 대응하는 int 값을 담은 (batch,) 형태의 Tensor
	"""
	return torch.argmax(one_hot_tensor, dim=1)
