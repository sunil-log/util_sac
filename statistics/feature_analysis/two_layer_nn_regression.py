
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""

import numpy as np

from torch.utils.data import DataLoader, Dataset


import torch
import torch.nn as nn
import torch.nn.functional as F

from util_sac.data.epoch_metric_tracker import metric_tracker
from util_sac.data.print_array_info import print_array_info
from util_sac.pytorch.trainer.trainer import BaseTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TwoLayerNetTrainer(BaseTrainer):

	def one_step(self, batch, epoch):
		X, y = batch
		"""
		X		  PyTorch Tensor	   (32, 64)					 8.00 KB torch.float32
		y		  PyTorch Tensor	   (32,)					   256.00 B torch.int64
		"""

		X = X.to(device)
		y = y.to(device)

		# 필요하다면 (batch,) -> (batch, 1)
		y = y.unsqueeze(-1)

		# forward
		y_hat = self.model(X)  # (batch, 1)
		loss = self.criterion(y_hat, y)  # nn.MSELoss 등

		self.loss_collector.update(loss=loss.item())
		self.data_collector.update(y_hat=y_hat, y=y)
		return loss



class TwoLayerNet(nn.Module):
	"""
	PyTorch를 사용한 2-Layer Neural Network 클래스.
	single scalar를 output하는 예시(회귀).
	"""
	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)  # <-- 변경

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)  # (batch, 1)
		return x


class CustomDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]




def train_nn_regression(
		X_train, y_train,
		X_test, y_test,
		n_epoch=100, batch_size=32, lr=1e-4
):

	"""
	print_array_info({ "X_train": X_train, "y_train": y_train,
					   "X_test": X_test, "y_test": y_test, })
	X_train	NumPy Array		  (151, 3)					 3.54 KB int64
	y_train	NumPy Array		  (151,)					   1.18 KB float64
	X_test	 NumPy Array		  (65, 3)					  1.52 KB int64
	y_test	 NumPy Array		  (65,)					   520.00 B float64
	"""

	# cast X, y to torch.Tensor float32
	X_train = torch.tensor(X_train, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.float32)


	# dataloader
	train_dataset = CustomDataset(X_train, y_train)
	test_dataset = CustomDataset(X_test, y_test)
	dataloaders = {
		'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
		'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	}

	# model
	# model
	model = TwoLayerNet(
		input_size=X_train.shape[1],
		hidden_size=16,
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()

	# trainer
	trainer = TwoLayerNetTrainer(
		model=model,
		dataloaders=dataloaders,
		optimizer=optimizer,
		criterion=criterion,
		n_epoch=n_epoch,
	)

	# train
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):
		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		"""
		print_array_info(train_data)
		logits	 PyTorch Tensor	   (1442, 2)				   11.27 KB torch.float32
		y		  PyTorch Tensor	   (1442,)					 11.27 KB torch.int64
		"""

		mt.update(epoch, **train_loss, **test_loss)
		mt.print_latest()

	df = mt.generate_df()
	return model, df



if __name__ == "__main__":

	# concatenate
	X = np.concatenate(list_eeg, axis=0)
	y = np.concatenate(list_label, axis=0)
	feature_names = [f'feature_{i}' for i in range(X.shape[1])]
	"""
	X		NumPy Array		  (2061, 64)				 515.25 KB float32
	y		NumPy Array		  (2061,)					 16.10 KB int64
	"""

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=42
	)

	res = train_nn_regression(X_train, y_train, X_test, y_test, n_epoch=100, lr=1e-4)
