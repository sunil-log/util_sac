

import torch

def one_hot_to_int(one_hot_tensor: torch.Tensor) -> torch.Tensor:
	"""
	(batch, feature) 형태의 one-hot Tensor를 해당하는 int로 변환하는 함수.

	Args:
		one_hot_tensor (torch.Tensor): (batch, feature) 형태의 one-hot Tensor

	Returns:
		torch.Tensor: 각 row마다 one-hot에 대응하는 int 값을 담은 (batch,) 형태의 Tensor
	"""
	return torch.argmax(one_hot_tensor, dim=1)
