
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""

import torch

def apply_mask_dict(tensor_dict: dict, mask: torch.Tensor) -> dict:
	"""
	하나의 함수가 (B,), (B,1), (B,T) 형태의 마스크 모두를 지원.

	Args:
		tensor_dict (dict): {key: Tensor} 형태의 dictionary
			- key: str
			- value: torch.Tensor
			  - 1차원 마스크일 경우 : value.shape[0] = B
			  - 2차원 마스크일 경우 : value.shape[:2] = (B, T)
		mask (torch.Tensor): 지원되는 마스크 형태
			- (B,) 또는 (B,1)  => 1차원으로 처리
			- (B,T)			=> 2차원으로 처리

	Returns:
		dict: mask가 True인 위치만 필터링된 tensor_dict

	주의:
		- 2차원 (B,T) 마스크 적용 시, 결과 shape는 (num_valid, ...)가 됩니다.
		  즉 B, T 두 축을 펼친 뒤 남은 row들만 가져오기 때문입니다.
		- (B,1) 마스크는 (B,)로 펼쳐서 동일하게 처리합니다.
	"""

	out_dict = {}
	ndim = mask.ndim

	# ----------------
	# Case 1) 1차원 마스크 (B,) or (B,1)
	# ----------------
	if ndim == 1 or (ndim == 2 and mask.shape[1] == 1):
		# (B,1)인 경우 squeeze
		if ndim == 2:
			mask = mask.view(-1)  # (B,1) -> (B,)

		# 이제 mask.shape = (B,)
		# 바로 인덱싱
		for k, v in tensor_dict.items():
			# v.shape[0]이 B와 같다고 가정
			out_dict[k] = v[mask]

	# ----------------
	# Case 2) 2차원 마스크 (B,T)
	# ----------------
	elif ndim == 2:
		B, T = mask.shape

		# 2D mask를 1차원으로 Flatten
		mask_1d = mask.view(-1)  # (B*T,)

		for k, v in tensor_dict.items():
			# 예: v.shape = (B, T, ...) 여야 함
			if v.shape[0] != B or v.shape[1] != T:
				raise ValueError(
					f"[apply_mask_dict] 텐서 {k}의 앞 두 차원 {tuple(v.shape[:2])}이 "
					f"마스크 (B,T)=({B},{T})와 일치하지 않습니다."
				)
			# (B, T, ...) -> (B*T, ...)
			flat_v = v.view(-1, *v.shape[2:])  # 첫 두 축을 펼치기
			# mask 적용
			out_dict[k] = flat_v[mask_1d]

	else:
		raise ValueError(
			f"[apply_mask_dict] 지원하지 않는 mask 차원입니다: mask.ndim={ndim}"
		)

	return out_dict




def main():
	pass

if __name__ == "__main__":
	main()
