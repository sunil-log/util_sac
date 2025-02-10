import torch
import numpy as np


def create_mask_single(
		data: torch.Tensor,	  # 필터 조건을 확인할 텐서 (예: stage, class_rbd 등)
		target,				  # 단일 타겟 (int 또는 1D/2D 텐서 가능, list나 numpy로도 가능)
		dim: int = -1,		  # 어느 차원에서 비교할지
		match_mode: str = 'all'  # 'all' or 'any'
) -> torch.Tensor:
	"""
	data.shape = (..., dim_size)
	target.shape = (dim_size,) 또는 단일 스칼라 int 가능
	match_mode 이 'all' 이면 전부 일치해야 True, 'any'면 하나라도 일치하면 True.
	반환값은 data에서 비교 차원을 줄인 boolean mask.
	list나 numpy 타입의 target은 data의 device와 동일하도록 옮긴다.
	"""

	# target 이 list 나 numpy 인 경우 -> 동일 device 의 텐서로 변환
	if isinstance(target, (list, np.ndarray)):
		target = torch.tensor(target, device=data.device)

	# case 1) target 이 int 인 경우
	if isinstance(target, int):
		# (data == target) -> bool 텐서
		mask = (data == target)
		return mask

	# case 2) target 이 텐서(또는 list에서 변환된 텐서)인 경우
	eq = (data == target)  # shape 동일, bool 텐서

	if match_mode == 'all':
		return eq.all(dim=dim)
	elif match_mode == 'any':
		return eq.any(dim=dim)
	else:
		raise ValueError("match_mode must be either 'all' or 'any'")



def create_mask_multi_targets(
	data: torch.Tensor,
	targets,				# [ [0,1,0,0,0], [0,0,1,0,0], ... ] 처럼 여러 개
	dim: int = -1,
	match_mode: str = 'all' # 보통 'all'을 많이 씀. 'any'가 필요한 경우도 가능
) -> torch.Tensor:

	"""
	주어진 data Tensor와 여러 targets를 비교하여 최종적으로 boolean mask를 생성한다.
	내부적으로 ... 함수를 반복 호출하고, 생성된 mask들을 OR 연산으로 통합한다.

	Args:
		data (torch.Tensor):
			비교 대상이 되는 Tensor. 예: stage 정보, class_rbd 등.
		targets:
			다수의 target을 담은 list. 각 요소는 int, list, 혹은 Tensor가 될 수 있으며,
			예를 들어 [[0,1,0,0,0], [0,0,1,0,0]] 형태처럼 여러 target 집합을 지정 가능하다.
		dim (int, optional):
			match를 수행할 차원. 기본값은 -1이다.
		match_mode (str, optional):
			'all' 혹은 'any' 중 하나를 지정. 'all'은 해당 차원의 모든 값이 target과 일치해야 True가 되고,
			'any'는 하나라도 일치하면 True가 된다. 기본값은 'all'이다.

	Returns:
		torch.Tensor:
			targets 중 하나라도 조건을 만족하는 위치에 True가 설정된 boolean mask Tensor.
	"""

	# targets 중 하나라도 맞으면 True 가 되도록 OR 연산 누적
	final_mask = None
	for t in targets:
		mask = create_mask_single(data, target=t, dim=dim, match_mode=match_mode)
		if final_mask is None:
			final_mask = mask
		else:
			final_mask = final_mask | mask  # 논리 OR
	return final_mask



def create_mask(
		data: torch.Tensor,
		target,
		dim: int = -1,
		match_mode: str = 'all',
		multi: bool = False
) -> torch.Tensor:
	"""
	data와 target(또는 targets)을 비교해 boolean mask를 생성한다.
	multi=True이면 복수의 target이 주어졌다고 간주하여 create_mask_multi_targets를 호출,
	그렇지 않으면 create_mask를 호출한다.

	Args:
		data (torch.Tensor):
			비교 대상이 되는 Tensor
		target:
			단일 target(int, list, Tensor 등) 혹은 복수 target(list, tuple, 등)
		dim (int, optional):
			비교할 차원. 기본값은 -1
		match_mode (str, optional):
			'all' 혹은 'any'; 보통은 모든 값이 일치해야 하므로 'all'을 사용한다.
		multi (bool, optional):
			복수 target인지 여부. 기본값 False
	"""
	if multi:
		return create_mask_multi_targets(data, target, dim=dim, match_mode=match_mode)
	else:
		return create_mask_single(data, target, dim=dim, match_mode=match_mode)




def apply_mask_dict(
	tensor_dict: dict,
	mask: torch.Tensor
) -> dict:
	"""
	tensor_dict: { key: torch.Tensor } 형태의 dictionary
	mask: 1D boolean mask Tensor (shape: (N,))
		- True/False 값을 사용해 필터링할 때 사용
	Returns:
		필터링된 dictionary (key는 그대로, 값은 mask를 적용한 Tensor)

	2D mask:
		2D mask (e.g. batch. epoch) 는 함부로 적용하면 tensor 구조가 깨질 수 있어서 구현하지 않는다.
		다만 create_mask 함수에서 dim을 지정하면 2D mask 를 생성하는 것은 가능하다.
	"""
	out_dict = {}
	for k, v in tensor_dict.items():
		out_dict[k] = v[mask]
	return out_dict




if __name__ == '__main__':

	# from util_sac.data.print_array_info import print_array_info
	# from util_sac.pytorch.filter_tensor import filter_data_by_target

	# 예시 data 생성: [B, ts, feature] = [6, 10, 3]
	x1 = torch.randn(6, 10, 3)
	x2 = torch.randn(6, 2, 3)

	# 예시 label 생성: [B, 5] = [6, 5]
	label = torch.tensor([
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[1, 0, 0, 0, 0],
		[0, 1, 0, 0, 0],
		[0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0]
	], dtype=torch.int64)

	data = {"x1": x1, "x2": x2, "label": label}
	"""
	target_info = {
		"key": "label",
		"data": [0, 1, 0, 0, 0],
		"dim": -1,
		"multi": False
	}
	"""
	target_info = {
		"key": "label",
		"data": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
		"dim": -1,
		"multi": True
	}


	filtered_data = filter_data_by_target(data, target_info)

	"""
	print_array_info(filtered_data)
	print(filtered_data["label"])
	
	x1         PyTorch Tensor       (4, 10, 3)                  480.00 B torch.float32
	x2         PyTorch Tensor       (4, 2, 3)                    96.00 B torch.float32
	label      PyTorch Tensor       (4, 5)                      160.00 B torch.int64

	tensor([[0, 1, 0, 0, 0],
	        [0, 0, 1, 0, 0],
	        [0, 1, 0, 0, 0],
	        [0, 1, 0, 0, 0]])
	"""
