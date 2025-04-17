
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

@author: sac
"""


import numpy as np
import pandas as pd

from util_sac.pytorch.print_array import print_array_info
from util_sac.pytorch.dataloader.to_tensor_device import move_dict_tensors_to_device
from util_sac.pytorch.trainer.trainer import BaseTrainer


class NewTrainer(BaseTrainer):

	def __init__(self, **kwargs):
		# BaseTrainer를 상속받으며, my_param 파라미터를 추가로 받는 클래스입니다.
		super().__init__(**kwargs)

		# 여기서 검증이나 로직이 필요하면 명시적으로 꺼낸다.
		if not hasattr(self, 'args'):
			raise ValueError("args가 없습니다. NewTrainer를 사용할 때는 반드시 args를 전달해야 합니다.")
		self.args = kwargs['args']


	def one_step(self, batch, epoch):

		"""
		"""

		# move to device
		d = move_dict_tensors_to_device(batch, self.device)
		"""
		path       PyTorch Tensor       (16,)                       128.00 B torch.int64
		W단어        PyTorch Tensor       (16, 9, 16)                 18.00 KB torch.int64
		W선택단어      PyTorch Tensor       (16, 9, 1)                   1.12 KB torch.int64
		W응답시간      PyTorch Tensor       (16, 9, 1)                  576.00 B torch.float32
		W제시자극      PyTorch Tensor       (16, 9, 1)                   1.12 KB torch.int64
		WDwelling  PyTorch Tensor       (16, 9, 16)                  9.00 KB torch.float32
		WEvalWord  PyTorch Tensor       (16, 9, 16)                 18.00 KB torch.int64
		WVisits    PyTorch Tensor       (16, 9, 16)                  9.00 KB torch.float32
		y          PyTorch Tensor       (16,)                       128.00 B torch.int64
		demographics PyTorch Tensor       (16, 8)                     512.00 B torch.float32
		iat_valid_mask PyTorch Tensor       (16, 2, 40, 1)              10.00 KB torch.int64
		iat_emb    PyTorch Tensor       (16, 2, 40, 8)              40.00 KB torch.float32
		"""


		# forward
		logits = self.model(d)
		loss = self.criterion(logits, d['y'])
		"""
		logits     PyTorch Tensor       (32, 10)                     1.25 KB torch.float32
		loss       PyTorch Tensor       ()                            4.00 B torch.float32
		"""

		# collect loss
		self.loss_collector.update(
			loss=loss.item(),
		)

		# collect test trials
		self.data_collector.update(
			logits=logits,         # logits 가 있어야 metric 계산 가능
			y=d['y'],             # y 가 있어야 metric 계산 가능
		)

		return loss