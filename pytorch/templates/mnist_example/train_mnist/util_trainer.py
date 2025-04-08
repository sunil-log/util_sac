
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

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
		x          PyTorch Tensor       (32, 1, 784)                98.00 KB torch.float32
		y          PyTorch Tensor       (32,)                       256.00 B torch.int64
		"""

		# forward
		logits = self.model(d['x'])
		loss = self.criterion(logits, d['y'])
		"""
		logits     PyTorch Tensor       (32, 10)                     1.25 KB torch.float32
		loss       PyTorch Tensor       ()                            4.00 B torch.float32
		"""

		# collect loss
		self.loss_collector.update(
			loss=loss.item(),
		)

		# collect test data
		self.data_collector.update(
			logits=logits,         # logits 가 있어야 metric 계산 가능
			y=d['y'],             # y 가 있어야 metric 계산 가능
		)

		return loss