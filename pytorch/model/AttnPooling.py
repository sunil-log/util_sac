

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttnPoolingWithMask(nn.Module):
	"""
	(batch, seq, input_dim) 형태의 입력 x와 (batch, seq) 형태의 mask를 받아 multi-head attention pooling을 수행한 뒤,
	(batch, output_dim) 형태로 요약 벡터(또는 class logits)를 산출한다.

	Parameters
	----------
	input_dim : int
	    입력 벡터의 차원 (예: 32).
	hidden_dim : int
	    Attention 연산에서 사용하는 내부 차원 (예: 64). num_heads의 배수여야 한다.
	num_heads : int
	    병렬로 Attention을 수행할 head의 개수 (예: 4).
	output_dim : int
	    최종 산출되는 벡터(또는 class logits)의 차원 (예: 2).
	dropout_p : float
	    Attention weight에 적용할 Dropout의 비율 (예: 0.1).

	Inputs
	------
	x : torch.Tensor
	    (batch, seq, input_dim) 형태의 입력. 시계열이나 문장 등 순차 데이터로 가정.
	mask : torch.Tensor or None
	    (batch, seq) 형태의 binary mask. 1이면 해당 위치를 사용(keep)하고, 0이면 무시(ignore)한다.
	    기본값(None)이면 전체 시퀀스를 사용한다.

	    ※ 주의: Transformer에서 사용하는 mask는 보통 1이 "ignore," 0이 "keep"인 반면,
	      본 모듈의 mask는 1이 "keep," 0이 "ignore"로 해석되므로 혼동에 유의한다.

	Returns
	-------
	out : torch.Tensor
	    (batch, output_dim) 형태의 최종 출력 벡터.
	att_weights : torch.Tensor
	    (batch, seq, num_heads, 1) 형태의 Attention 가중치.
	    seq 차원에 대해 softmax가 적용된 결과이며, 각 head별로 입력 시퀀스 위치에 대한 가중치를 나타낸다.

	Notes
	-----
	1) Key와 Value는 입력 x에 대해 별도의 linear 변환을 거쳐 계산한다.
	2) Query는 (num_heads, q_dim) 형태의 학습 가능한 파라미터로 구성한다.
	3) 최종적으로 multi-head로 구한 벡터들을 concat한 뒤, linear를 통과시켜 (batch, output_dim)을 산출한다.
	4) mask가 주어지면, mask가 0인 위치는 Attention score에 -inf를 더해 softmax에서 제외한다.
	"""

	def __init__(self,
				 input_dim=32,
				 hidden_dim=64,
				 num_heads=4,
				 output_dim=2,
				 dropout_p=0.1
				 ):
		super().__init__()
		self.num_heads = num_heads
		assert hidden_dim % num_heads == 0, "hidden_dim이 num_heads로 나누어 떨어져야 한다."
		self.q_dim = hidden_dim // num_heads

		self.key_net = nn.Linear(input_dim, hidden_dim)
		self.value_net = nn.Linear(input_dim, hidden_dim)
		self.query = nn.Parameter(torch.randn(num_heads, self.q_dim))

		self.fc = nn.Linear(hidden_dim, output_dim)
		self.attn_dropout = nn.Dropout(dropout_p)



	def flatten_batch(self, x, mask=None):
		"""
		x가 (B, T, D) 형태를 넘어서는 차원 구조를 가지면,
		앞쪽 차원을 전부 곱해 하나로 합친 뒤 (new_B, T, D) 형태로 바꾼다.

		Returns
		-------
		x_flat : torch.Tensor
			Flatten된 x
		mask_flat : torch.Tensor or None
			Flatten된 mask (있으면)
		do_unflatten : bool
			추후에 unflatten이 필요한지 여부
		saved_shapes : tuple
			복원에 필요한 원본 shape 정보
		"""
		# x가 (batch, seq, dim)보다 차원이 많아야 flatten이 필요함
		if x.dim() <= 3:
			return x, mask, False, (x.shape, mask.shape if mask is not None else None)

		original_shape_x = x.shape  # 예: (B1, B2, ..., T, D)
		T = x.shape[-2]
		D = x.shape[-1]

		# 앞쪽 차원을 전부 곱해 new_B를 구함
		new_B = 1
		for s in x.shape[:-2]:
			new_B *= s

		# x를 (new_B, T, D)로 변환
		x_flat = x.view(new_B, T, D)

		# mask도 동일하게 변환
		original_shape_mask = None
		mask_flat = None
		if mask is not None:
			original_shape_mask = mask.shape  # 예: (B1, B2, ..., T)
			mask_flat = mask.view(new_B, T)

		return x_flat, mask_flat, True, (original_shape_x, original_shape_mask)



	def unflatten_batch(self, out, att_weights, do_unflatten, saved_shapes):
		"""
		flatten_batch에서 flatten된 x, mask에 대해, 연산 결과 out, att_weights를 다시 원본 차원으로 복원.

		Parameters
		----------
		out : torch.Tensor
			(new_B, output_dim) 형태의 연산 결과
		att_weights : torch.Tensor
			(new_B, T, num_heads, 1) 형태의 Attention Weight
		do_unflatten : bool
			flatten_batch에서 실제 flatten이 일어났는지 여부
		saved_shapes : tuple
			(original_shape_x, original_shape_mask) 형태의 복원 정보

		Returns
		-------
		out_unflat : torch.Tensor
			원본 batch 차원으로 복원된 out
		att_weights_unflat : torch.Tensor
			원본 batch 차원으로 복원된 att_weights
		"""
		if not do_unflatten:
			# flatten이 일어나지 않았다면 그대로 반환
			return out, att_weights

		original_shape_x, _ = saved_shapes
		# x의 원본 shape: (B1, B2, ..., T, D)

		# out shape 복원
		# out: (new_B, output_dim) -> (B1, B2, ..., output_dim)
		out_unflat = out.view(*original_shape_x[:-2], -1)

		# att_weights shape 복원
		# att_weights: (new_B, T, num_heads, 1) -> (B1, B2, ..., T, num_heads, 1)
		T = original_shape_x[-2]
		att_weights_unflat = att_weights.view(*original_shape_x[:-2], T, self.num_heads, 1)

		return out_unflat, att_weights_unflat

	def forward(self, x, mask=None):
		# 1) flatten
		x_flat, mask_flat, do_unflatten, saved_shapes = self.flatten_batch(x, mask)

		B, S, _ = x_flat.shape

		# 2) K, V 계산
		K = self.key_net(x_flat)	# (B, S, hidden_dim)
		V = self.value_net(x_flat)  # (B, S, hidden_dim)

		K = K.view(B, S, self.num_heads, self.q_dim)
		V = V.view(B, S, self.num_heads, self.q_dim)

		# 3) s = dot(Q, K)
		s = torch.einsum('bsnh,nh->bsn', K, self.query)  # (B, S, num_heads)
		s = s / (self.q_dim ** 0.5)

		# 4) mask가 있으면 적용
		if mask_flat is not None:
			s = s.masked_fill(mask_flat.unsqueeze(-1) == 0, float('-inf'))

		# 5) softmax + dropout
		att_weights = F.softmax(s, dim=1)  # (B, S, num_heads)
		att_weights = self.attn_dropout(att_weights)
		att_weights = att_weights.unsqueeze(-1)  # (B, S, num_heads, 1)

		# 6) weighted sum
		weighted_V = V * att_weights
		z = weighted_V.sum(dim=1)  # (B, num_heads, q_dim)
		z = z.view(B, -1)		  # (B, hidden_dim)

		# 7) 최종 FC
		out = self.fc(z)  # (B, output_dim)

		# 8) unflatten
		out_unflat, att_weights_unflat = self.unflatten_batch(out, att_weights, do_unflatten, saved_shapes)

		return out_unflat, att_weights_unflat


if __name__ == "__main__":
	model = MultiHeadAttnPoolingWithMask(
		input_dim=32,
		hidden_dim=64,
		num_heads=4,
		output_dim=2
	)

	# 임의 데이터
	x = torch.randn(8, 10, 32)  # (batch=8, seq=10, dim=32)

	# 마스크 예시: 랜덤하게 0/1 부여
	# 여기서는 50% 확률로 1, 50% 확률로 0
	# 실제로는 padding 위치에 0, 유효 위치에 1을 넣음
	mask = (torch.rand(8, 10) > 0.5).int()

	# forward
	y = model(x, mask=mask)
	print(y.shape)  # (8, 2)