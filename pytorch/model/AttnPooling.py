

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttnPoolingWithMask(nn.Module):
	"""
	(batch, seq, input_dim) 형태의 입력 x와 (batch, seq) 형태의 mask를 받아
	multi-head attention pooling을 수행한 뒤,
	(batch, output_dim) 형태로 요약 벡터(또는 class logits)를 산출한다.
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
		# q_dim = (hidden_dim / num_heads)
		assert hidden_dim % num_heads == 0, "hidden_dim이 num_heads로 나누어 떨어져야 한다."
		self.q_dim = hidden_dim // num_heads

		# 입력을 Key로 변환 (linear): (input_dim) -> (hidden_dim)
		self.key_net = nn.Linear(input_dim, hidden_dim)

		# 입력을 Value로 변환 (linear): (input_dim) -> (hidden_dim)
		self.value_net = nn.Linear(input_dim, hidden_dim)

		# 각 head마다 learnable Query를 둔다: (num_heads, q_dim)
		self.query = nn.Parameter(torch.randn(num_heads, self.q_dim))

		# 모든 헤드를 concat한 후 (hidden_dim) -> (output_dim) 으로 이어지는 FC
		self.fc = nn.Linear(hidden_dim, output_dim)

		# Attention weight에 쓸 Dropout
		self.attn_dropout = nn.Dropout(dropout_p)



	def forward(self, x, mask=None):
		"""
		x: (batch, seq, input_dim)
		mask: (batch, seq) with 1/True=keep, 0/False=ignore. 기본값=None
		return: (batch, output_dim)
		"""
		B, S, _ = x.shape

		# --- 1) K, V 구하기
		#		(batch, seq, hidden_dim)
		K = self.key_net(x)
		V = self.value_net(x)

		# (batch, seq, num_heads, q_dim) 로 reshape
		K = K.view(B, S, self.num_heads, self.q_dim)
		V = V.view(B, S, self.num_heads, self.q_dim)

		# --- 2) attention score 구하기 (dot product)
		# K: (batch, seq, num_heads, q_dim)
		# Q: (num_heads, q_dim)
		# s: (batch, seq, num_heads)
		# s_{b,t,h} = Q_h^T K_{b,t,h}
		s = torch.einsum('bsnh,nh->bsn', K, self.query)

		# --- 3) scale
		s = s / (self.q_dim ** 0.5)  # sqrt(q_dim)로 나눔

		# --- 4) mask 적용
		# mask: (batch, seq). 1이면 살리고, 0이면 무시
		# s: (batch, seq, num_heads)
		if mask is not None:
			# mask가 0인 지점 = 무시해야 할 지점
			# => 해당 위치의 score를 매우 큰 음수로 만들어 softmax에서 0이 되게 한다.
			s = s.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

		# --- 5) softmax
		att_weights = F.softmax(s, dim=1)  # (batch, seq, num_heads)

		# Attention weight에 Dropout 적용
		att_weights = self.attn_dropout(att_weights)

		# --- 6) weighted sum
		# V: (batch, seq, num_heads, q_dim)
		# att_weights: (batch, seq, num_heads)
		# broadcast 위해 unsqueeze(-1)
		att_weights = att_weights.unsqueeze(-1)  # (batch, seq, num_heads, 1)
		weighted_V = V * att_weights  # (batch, seq, num_heads, q_dim)

		# seq 차원 합산 -> (batch, num_heads, q_dim)
		z = weighted_V.sum(dim=1)

		# (batch, num_heads, q_dim)을 concat -> (batch, hidden_dim)
		z = z.view(B, -1)  # (batch, hidden_dim)

		# 최종 출력
		out = self.fc(z)  # (batch, output_dim)
		return out



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