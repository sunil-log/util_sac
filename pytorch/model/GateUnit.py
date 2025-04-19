import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class GatedMultimodalUnit(nn.Module):
	"""
	Purpose
	-------
	서로 다른 modality의 표현을 gating‑기반으로 융합(fusion)하는 **Gated Multimodal Unit (GMU)** 구현이다.
	각 modality별 임베딩을 선형 변환 + tanh 로 비선형화한 후,
	동일 차원의 gate(sigmoid)를 곱해 정보 유입을 조절하고, 모든 gated 벡터를 합산하여 단일 표현을 얻는다.

	Parameters
	----------
	modalities : List[str]
	    처리할 modality 이름 목록. 입력 dict의 키와 일치해야 한다.
	in_dim : int
	    각 modality 입력 벡터의 차원 D.
	hidden_dim : int, optional
	    내부 변환 차원 H. 지정하지 않으면 in_dim과 동일하게 설정된다.
	p_dropout : float, default=0.2
	    gating 결과(h ⊙ g)에 적용할 dropout 확률.

	Input
	-----
	x : Dict[str, torch.Tensor]
	    `{modality: tensor(B, D)}` 형태의 배치 입력.
	    `B`는 batch size, `D = in_dim`.

	Output
	------
	torch.Tensor
	    융합된 표현 tensor(B, H),  `H = hidden_dim or in_dim`.

	Example
	-------
	>>> modalities = ["text", "image", "audio"]
	>>> gmu = GatedMultimodalUnit(modalities, in_dim=128, hidden_dim=256, p_dropout=0.2)
	>>> batch = {m: torch.randn(32, 128) for m in modalities}   # (batch=32)
	>>> fused = gmu(batch)                                       # tensor(32, 256)
	>>> fused.shape
	torch.Size([32, 256])
	"""

	def __init__(
		self,
		modalities: List[str],
		in_dim: int,
		hidden_dim: int | None = None,
		p_dropout: float = 0.2,
	):
		super().__init__()
		self.modalities = modalities
		H = hidden_dim or in_dim

		self.fc_tanh = nn.ModuleDict({m: nn.Linear(in_dim, H) for m in modalities})
		self.fc_gate = nn.ModuleDict({m: nn.Linear(in_dim, H) for m in modalities})

		# gating 이후 dropout만 유지
		self.drop_mid = nn.Dropout(p_dropout)

	def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
		"""
		x : {modality: tensor(B, D)}
		return : tensor(B, H)
		"""
		z_list = []
		for m in self.modalities:
			h_m = torch.tanh(self.fc_tanh[m](x[m]))
			g_m = torch.sigmoid(self.fc_gate[m](x[m]))
			z_m = self.drop_mid(h_m * g_m)
			z_list.append(z_m)

		fused = torch.stack(z_list, dim=0).sum(dim=0)
		return fused
