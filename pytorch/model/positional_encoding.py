
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""

import torch
import math


def positional_encoding(max_len=1000, d_model=64):
	# pos: [0, 1, 2, ..., max_len-1] (shape: (max_len, 1))
	pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

	# 각 차원별로 사용할 frequency(denominator) 계산 (shape: (1, d_model/2))
	div_term = torch.exp(
		torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
	)

	# (max_len, d_model) 크기의 zero tensor를 준비
	pe = torch.zeros(max_len, d_model)

	# 짝수 index(dim)에는 sin, 홀수 index(dim)에는 cos를 이용해 값 채우기
	pe[:, 0::2] = torch.sin(pos * div_term)
	pe[:, 1::2] = torch.cos(pos * div_term)

	# Shape을 (1, max_len, d_model)로 만들어 batch 단위 연산에서 더 쉽게 사용
	pe = pe.unsqueeze(0)
	return pe
