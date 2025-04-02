
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

from torch.utils.data import Dataset, DataLoader


def create_dataloaders(data, batch_size=32, shuffle=True):
	"""
	주어진 데이터를 DataLoader로 변환합니다.
	:param data: 변환할 데이터
	data = {
		'train': data_train,
		'valid': data_valid,
		'test': data_test
	}
	:param batch_size: 배치 크기
	:param shuffle: 데이터를 섞을지 여부
	:return: DataLoader 객체
	"""


	train_loader = DataLoader(
		TensorDataset(data['train']),
		batch_size=batch_size,
		shuffle=shuffle
	)
	valid_loader = DataLoader(
		TensorDataset(data['valid']),
		batch_size=batch_size,
		shuffle=False
	)
	test_loader = DataLoader(
		TensorDataset(data['test']),
		batch_size=batch_size,
		shuffle=False
	)

	return {
		'train': train_loader,
		'valid': valid_loader,
		'test': test_loader
	}




class TensorDataset(Dataset):
	def __init__(self, tensor_data):
		self.tensor_data = tensor_data
		self.length = len(next(iter(tensor_data.values())))
		"""
		self.length = 4
		self.tensor_data = 
			Key        Type            Shape                    Memory Dtype     
			----------------------------------------------------------------------
			REM_emg    PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
			REM_mask   PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
			...
		"""

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		return {key: value[idx] for key, value in self.tensor_data.items()}




def main():
	pass

if __name__ == "__main__":
	main()
