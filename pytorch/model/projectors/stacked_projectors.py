
# -*- coding: utf-8 -*-
"""
Created on  Apr 02 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.sys.date_format import add_timestamp_to_string
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette

import torch
import torch.nn as nn


class OneLayerProjector(nn.Module):
	""" 단일 Linear Layer """
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.linear = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		return self.linear(x)


class TwoLayerProjector(nn.Module):
	""" 2-Layer (Linear -> ReLU -> Linear) """
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.ReLU(),
			nn.Linear(out_dim, out_dim)
		)

	def forward(self, x):
		return self.layers(x)


class ResidualTwoLayerProjector(nn.Module):
	"""
	2-Layer Perceptron (Linear + ReLU) 구조 위에 Residual Connection, BatchNorm, Dropout을 적용하는 예시이다.
	입력 Feature Dimension과 출력 Feature Dimension이 다른 경우를 대비하여 Projection Layer를 사용한다.

	내부적으로 입력 텐서의 마지막 차원만을 Feature Dimension으로 보고 나머지 차원은 Batch Dimension으로 처리한다.
	예를 들어, 입력이 (a, b, c, d) 형태라면 (a, b, c)는 Batch 차원으로 간주되고, 마지막 차원인 d가 Feature 차원이다.

	Args:
	    in_dim (int): 입력 Feature의 Dimension이다.
	    hidden_dim (int): 첫 번째 Linear Layer의 출력 Dimension이다.
	    out_dim (int): 두 번째 Linear Layer의 출력 Dimension이자 전체 모델의 최종 출력 Dimension이다.
	    p_dropout (float, optional): Dropout 확률이다. 기본값은 0.5이다.

	Forward:
	    x (Tensor): (BatchSize, in_dim) 형태의 입력 Tensor이다. (BatchSize는 1개 이상의 Batch 차원으로 구성될 수 있다.)
	                 예: (a, b, c, d) 형태라면, (a, b, c)가 Batch 차원이고 마지막 d가 Feature 차원이다.

	Returns:
	    Tensor: (BatchSize, out_dim) 형태의 출력 Tensor이다. Residual Connection 적용 결과가 더해진다.
	"""
	def __init__(self, in_dim, hidden_dim, out_dim, p_dropout=0.5):
		super().__init__()
		self.linear1 = nn.Linear(in_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=p_dropout)
		self.linear2 = nn.Linear(hidden_dim, out_dim)

		# Projection for residual if input dimension != output dimension
		self.projection = None
		if in_dim != out_dim:
			self.projection = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		# Branch 1: 일반적인 2-Layer 처리
		out = self.linear1(x)
		# Batch 차원(예: (BatchSize, FeatureDim))에서 BatchNorm을 적용하기 위해 차원이 맞도록 transpose 필요할 수 있음
		out = self.relu(out)
		out = self.dropout(out)
		out = self.linear2(out)

		# Branch 2: Residual Connection
		if self.projection is not None:
			x = self.projection(x)
		out = out + x  # Skip Connection
		return out


class StackedProjector(nn.Module):
	"""
	여러 개의 ResidualTwoLayerProjector를 직렬로 연결하여 Feature 변환을 연속적으로 수행한다.
	내부적으로 입력 텐서의 마지막 차원만을 Feature Dimension으로 보고 나머지 차원은 Batch Dimension으로 처리한다.
	예를 들어, 입력이 (a, b, c, d) 형태라면 (a, b, c)는 Batch 차원으로 간주되고, 마지막 차원인 d가 Feature 차원이다.

	Args:
	    in_dim (int): 첫 번째 ResidualTwoLayerProjector의 입력 Feature Dimension이다.
	    hidden_dim (int): 각 ResidualTwoLayerProjector에서의 첫 번째 Linear Layer 출력 Dimension이다.
	    out_dim (int): 모델의 최종 출력 Dimension이며 모든 ResidualTwoLayerProjector의 두 번째 Linear Layer가 가진 출력 Dimension이다.
	    p_dropout (float, optional): 각 ResidualTwoLayerProjector 내 Dropout 확률이다. 기본값은 0.5이다.
	    num_stacks (int, optional): 직렬로 연결할 ResidualTwoLayerProjector의 개수이다. 기본값은 3이다.

	Forward:
	    x (Tensor): (BatchSize, in_dim) 형태의 입력 Tensor이다. (BatchSize는 1개 이상의 Batch 차원으로 구성될 수 있다.)
	                 예: (a, b, c, d) 형태라면, (a, b, c)가 Batch 차원이고 마지막 d가 Feature 차원이다.

	Returns:
	    Tensor: (BatchSize, out_dim) 형태의 출력 Tensor이다.
	            ResidualTwoLayerProjector가 순차적으로 적용된 결과를 반환한다.
	"""
	def __init__(self, in_dim, hidden_dim, out_dim, num_stacks=3, p_dropout=0.5):
		super().__init__()
		self.projectors = nn.ModuleList()
		for i in range(num_stacks):
			if i == 0:
				self.projectors.append(
					ResidualTwoLayerProjector(in_dim, hidden_dim, out_dim, p_dropout)
				)
			else:
				# 두 번째 Projector부터는 이전 Projector의 출력 Dimension이 out_dim이므로
				self.projectors.append(
					ResidualTwoLayerProjector(out_dim, hidden_dim, out_dim, p_dropout)
				)

	def forward(self, x):
		for projector in self.projectors:
			x = projector(x)
		return x


if __name__ == "__main__":
	pass
