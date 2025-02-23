
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset


import torch
import torch.nn as nn
import torch.nn.functional as F



from util_sac.pandas.print_df import print_partial_markdown
from util_sac.data.print_array_info import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette
from util_sac.data.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trainer2 import BaseTrainer
from util_sac.metrics.multi_class_matrics import calculate_f1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TwoLayerNetTrainer(BaseTrainer):

	def one_step(self, batch, epoch):
		X, y = batch
		"""
		X          PyTorch Tensor       (32, 64)                     8.00 KB torch.float32
		y          PyTorch Tensor       (32,)                       256.00 B torch.int64
		"""

		X = X.to(device)
		y = y.to(device)

		# forward
		logits = self.model(X)
		loss = self.criterion(logits, y)

		self.loss_collector.update(
			loss=loss.item(),
		)

		self.data_collector.update(
			logits=logits,
			y=y,
		)
		return loss




class TwoLayerNet(nn.Module):
	"""
	PyTorch를 사용한 2-Layer Neural Network 클래스.

	Parameters
	----------
	input_size : int
		입력 데이터의 feature 개수입니다.
	hidden_size : int
		첫 번째 Layer의 output dimension 크기입니다.
	num_classes : int
		모델의 최종 output dimension 크기로, 예측해야 할 클래스의 개수를 의미합니다.
	"""

	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		"""
		Forward Pass 함수.

		입력 x는 (batch, feature) 형태입니다.
		두 개의 Linear Layer와 ReLU 비선형 함수를 통해
		최종적으로 (batch, num_classes) 형태의 출력을 반환합니다.

		Parameters
		----------
		x : torch.Tensor
			shape = (batch, input_size)

		Returns
		-------
		torch.Tensor
			shape = (batch, num_classes)
		"""
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class CustomDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]






def score_neural_net(
		X_train, y_train,
		X_test, y_test,
		feature_names,
):

	"""
	X_train    NumPy Array          (1442, 64)                 360.50 KB float32
	y_train    NumPy Array          (1442,)                     11.27 KB int64
	X_test     NumPy Array          (619, 64)                  154.75 KB float32
	y_test     NumPy Array          (619,)                       4.84 KB int64
	feature_names Other                <class 'list'>                   N/A N/A 
	"""

	# dataloader
	train_dataset = CustomDataset(X_train, y_train)
	test_dataset = CustomDataset(X_test, y_test)
	dataloaders = {
	    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
	    'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
	}



	# model
	n_class = len(np.unique(y_train))
	model = TwoLayerNet(
		input_size=X_train.shape[1],
		hidden_size=16,
		num_classes=n_class
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	# trainer
	trainer = TwoLayerNetTrainer(
		model=model,
		dataloaders=dataloaders,
		optimizer=optimizer,
		criterion=criterion,
		n_epoch=100,
	)

	# train
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):
		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		print_array_info(train_data)
		"""
		logits     PyTorch Tensor       (1442, 2)                   11.27 KB torch.float32
		y          PyTorch Tensor       (1442,)                     11.27 KB torch.int64
		"""

		f1_train = calculate_f1(train_data, name="train")
		f1_test = calculate_f1(test_data, name="test")


		mt.update(epoch, **train_loss, **test_loss, **f1_train, **f1_test)
		mt.print_latest()



if __name__ == "__main__":
	pass
