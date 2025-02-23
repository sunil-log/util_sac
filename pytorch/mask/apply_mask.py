import torch
import numpy as np


def apply_mask_1d(tensor_dict: dict, mask: torch.Tensor, device: torch.device) -> dict:
	"""
	1차원 mask (shape=(B,)) 또는 (B,1) 처리 전용 함수.
		예를 들어 (epoch, dim) 이 있는데,
		mask 가 (epoch) or (epoch, 1)= valid epoch 인 경우 사용

	1) mask가 numpy.ndarray라면 torch.Tensor로 변환
	2) mask가 0/1 (int/float)이면 bool로 변환 (mask != 0)
	3) dict 안의 value도 numpy -> tensor -> device
	4) 인덱싱 후 반환 (shape: (num_valid, ...))
	"""
	# mask를 device로 옮기기 전 numpy -> tensor 변환
	if isinstance(mask, np.ndarray):
		mask = torch.from_numpy(mask)

	# (B,1) -> (B,) 로 펼치기
	if mask.ndim == 2 and mask.shape[1] == 1:
		mask = mask.view(-1)

	# device로 이동
	mask = mask.to(device)
	# 0/1 형식이면 bool로 변환
	if mask.dtype != torch.bool:
		mask = (mask != 0)  # 0이면 False, 그 외(1,etc)이면 True

	out_dict = {}
	for k, v in tensor_dict.items():
		# v가 numpy array면 tensor로
		if isinstance(v, np.ndarray):
			v = torch.from_numpy(v)
		# device로 이동
		v = v.to(device)
		# 인덱싱
		out_dict[k] = v[mask].cpu()  # 여기서 .cpu()로 VRAM 절감
	return out_dict


def apply_mask_2d(tensor_dict: dict, mask: torch.Tensor, device: torch.device) -> dict:
	"""
	2차원 mask (shape=(B,T)) 처리 전용 함수.
		예를 들어 (subject, epoch, dim) 이 있는데,
		mask 가 (subject, epoch) = valid epoch 인 경우 사용

	1) mask가 numpy.ndarray라면 torch.Tensor로 변환
	2) mask가 0/1 (int/float)이면 bool로 변환 (mask != 0)
	3) dict 안의 value도 numpy -> tensor -> device
	4) (B,T) -> (B*T,)로 펼쳐서 인덱싱 후 반환
	   => 결과 shape: (num_valid, ...)
	"""
	# mask를 device로 옮기기 전 numpy -> tensor 변환
	if isinstance(mask, np.ndarray):
		mask = torch.from_numpy(mask)
	# device로 이동
	mask = mask.to(device)
	# 0/1 형식이면 bool로 변환
	if mask.dtype != torch.bool:
		mask = (mask != 0)

	B, T = mask.shape
	# 2D -> 1D
	mask_1d = mask.view(-1)  # (B*T,)

	out_dict = {}
	for k, v in tensor_dict.items():
		# v가 numpy array면 tensor로
		if isinstance(v, np.ndarray):
			v = torch.from_numpy(v)
		# device로 이동
		v = v.to(device)

		# shape 검사
		if v.shape[0] != B or v.shape[1] != T:
			raise ValueError(
				f"[apply_mask_2d] 텐서 {k} shape={v.shape[:2]} 이 (B,T)=({B},{T})와 맞지 않습니다."
			)

		# (B,T, ...) -> (B*T, ...)
		flat_v = v.view(-1, *v.shape[2:])
		out_dict[k] = flat_v[mask_1d].cpu()  # 여기서 .cpu()로 VRAM 절감
	return out_dict


def apply_mask_dict(tensor_dict: dict, mask, device: torch.device) -> dict:
	"""
	하나의 함수에서 1D/2D mask를 구분해서 처리.

		1차원 mask (shape=(B,)) 또는 (B,1)
			예를 들어 (epoch, dim) 이 있는데,
			mask 가 (epoch) or (epoch, 1)= valid epoch 인 경우 사용

		2차원 mask (shape=(B,T))
			예를 들어 (subject, epoch, dim) 이 있는데,
			mask 가 (subject, epoch) = valid epoch 인 경우 사용

	Args:
		tensor_dict (dict): {key: torch.Tensor or np.ndarray} 형태
			- 1D 마스크인 경우: v.shape[0] = B
			- 2D 마스크인 경우: v.shape[:2] = (B, T)
		mask (torch.Tensor or np.ndarray):
			- shape = (B,) or (B,1) => 1D case
			- shape = (B,T)		=> 2D case
			- 값이 bool이 아니고 0/1로 되어 있어도 사용 가능
		device (torch.device):
			- 연산 및 결과 텐서가 올라갈 디바이스 (cuda, cpu 등)
	Returns:
		dict: mask를 적용해 필터링된 dict

	동작 요약:
	  - 1D 마스크 => apply_mask_1d 호출
	  - 2D 마스크 => apply_mask_2d 호출
	"""

	# (B,1)도 1D 케이스로 간주, (B,T) => 2D
	# numpy 타입일 수도 있으므로 shape는 아래처럼 확인
	if isinstance(mask, np.ndarray):
		mask_shape = mask.shape
	else:
		mask_shape = mask.shape

	if len(mask_shape) == 1 or (len(mask_shape) == 2 and mask_shape[1] == 1):
		# 1D mask
		return apply_mask_1d(tensor_dict, mask, device)
	elif len(mask_shape) == 2:
		# 2D mask
		return apply_mask_2d(tensor_dict, mask, device)
	else:
		raise ValueError(
			f"[apply_mask_dict] 지원하지 않는 mask shape: {mask_shape}. "
			f"1D((B,) or (B,1)) 또는 2D((B,T))만 허용됩니다."
		)


if __name__ == "__main__":

	from util_sac.pytorch.mask.apply_mask import apply_mask_dict

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# 예: B=3, T=2
	my_dict = {
		"features": np.random.randn(3, 2, 5),	  # numpy (3,2,5)
		"labels":   torch.randint(0, 10, (3,2))	# torch.Tensor (3,2)
	}

	# 1) 1D 마스크 (B=3) - torch
	mask_1d = torch.tensor([True, False, True])   # shape=(3,)

	# 2) 2차원이지만 (B,1) -> 1D로 간주
	mask_2d_col = np.array([[1],[0],[1]])		 # numpy, shape=(3,1), 값=0/1

	# 3) 2D 마스크 (B=3, T=2) -> 값=0/1
	mask_2d = np.array([
		[1, 0],
		[0, 1],
		[1, 1]
	])

	# ------------------------
	# (1) 1D mask => shape=(2,2,5) / shape=(2,2)
	# ------------------------
	out1 = apply_mask_dict(my_dict, mask_1d, device)
	print("mask_1d 결과:")
	for k, v in out1.items():
		print(f"  {k} shape={v.shape}, device={v.device}")

	# ------------------------
	# (2) (B,1) mask => 실제로 1D로 처리
	# ------------------------
	out2 = apply_mask_dict(my_dict, mask_2d_col, device)
	print("\nmask_2d_col (B,1) 결과:")
	for k, v in out2.items():
		print(f"  {k} shape={v.shape}, device={v.device}")

	# ------------------------
	# (3) 2D mask (B,T)
	# ------------------------
	out3 = apply_mask_dict(my_dict, mask_2d, device)
	print("\nmask_2d (B,T) 결과:")
	for k, v in out3.items():
		print(f"  {k} shape={v.shape}, device={v.device}")
