import numpy as np
import torch

"""
이 모듈은 PyTorch Tensor 데이터를 특정 target 조건과 비교하여 boolean mask를 생성하고,
해당 mask를 사용해 dictionary 형태의 Tensor 데이터를 효율적으로 필터링하기 위한
유틸리티 함수들을 제공합니다.

주요 기능:
- create_mask_single:
    주어진 data Tensor와 하나의 target을 비교해 boolean mask를 생성합니다.
- create_mask_multi_targets:
    여러 target을 OR 연산으로 결합해 boolean mask를 생성합니다.
- create_mask:
    단일 또는 복수의 target에 대해 boolean mask를 생성할 수 있도록 관리하는 함수입니다.
- apply_mask_dict_1D:
    생성된 boolean mask를 dictionary 형태의 Tensor 데이터에 적용해 필터링을 수행합니다.

용례:
1) 단일 target에 대한 필터링
    예를 들어, data["label"]에 1이 포함된 row만 골라내고 싶다면:
    >>> mask = create_mask(data["label"], target=1, dim=-1)

2) 다중 target에 대한 필터링
    예를 들어, data["label"]에서 [0,1,0,0,0] 또는 [0,0,1,0,0]을 가진 row만 필터링하려면:
    >>> mask = create_mask(data["label"], target=[[0,1,0,0,0], [0,0,1,0,0]], dim=-1, multi=True)

이를 통해 다차원 Tensor에서 특정 조건('all' 혹은 'any')을 만족하는 요소들만 빠르게 추출하고,
추가 연산을 용이하게 수행할 수 있습니다.
"""



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



if __name__ == '__main__':

	# load data
	fn = './data/encoded__weight_4990__normalize_False.npz'
	data = np.load(fn)
	"""
	z_eeg      NumPy Array          (413, 1000, 12, 64)          1.18 GB float32
	z_eog      NumPy Array          (413, 1000, 12, 32)        604.98 MB float32
	z_emg      NumPy Array          (413, 1000, 12, 32)        604.98 MB float32
	z_mask     NumPy Array          (413, 1000)                  1.58 MB int32
	stage      NumPy Array          (413, 1000, 5)              15.75 MB int64
	class_rbd  NumPy Array          (413, 1000)                  3.15 MB int64
	class_pd   NumPy Array          (413, 1000)                  3.15 MB int64
	hospital   NumPy Array          (413, 1000, 5)              15.75 MB int64
	subject_id NumPy Array          (413, 1000, 413)             1.27 GB int64
	"""

	mask = data['z_mask'].copy()
	data = apply_mask_dict(data, mask, device)
	"""
	z_eeg      PyTorch Tensor       (273419, 12, 64)           801.03 MB torch.float32
	z_eog      PyTorch Tensor       (273419, 12, 32)           400.52 MB torch.float32
	z_emg      PyTorch Tensor       (273419, 12, 32)           400.52 MB torch.float32
	z_mask     PyTorch Tensor       (273419,)                    1.04 MB torch.int32
	stage      PyTorch Tensor       (273419, 5)                 10.43 MB torch.int64
	class_rbd  PyTorch Tensor       (273419,)                    2.09 MB torch.int64
	class_pd   PyTorch Tensor       (273419,)                    2.09 MB torch.int64
	hospital   PyTorch Tensor       (273419, 5)                 10.43 MB torch.int64
	subject_id PyTorch Tensor       (273419, 413)              861.53 MB torch.int64
	"""

	X = data['z_eeg']     # (273419, 12, 64)
	X = X.max(axis=1)[0]  # (273419, 64)
	y = create_mask(data["stage"], target=[0, 0, 0, 0, 1])  # torch.bool
	y = y.long()  # torch.int64
	"""
	X          PyTorch Tensor       (273419, 64)                66.75 MB torch.float32
	y          PyTorch Tensor       (273419,)                    2.09 MB torch.int64
	"""

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=42
	)


	res = train_neural_net(
		X_train, y_train,
		X_test, y_test,
		n_epoch=100, batch_size=128, lr=1e-4)
	print(res)

