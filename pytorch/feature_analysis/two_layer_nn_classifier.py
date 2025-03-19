
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
from util_sac.pytorch.trainer.trainer import BaseTrainer
from util_sac.pytorch.metrics import calculate_f1


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






def train_neural_net(
		X_train, y_train,
		X_test, y_test,
		n_epoch=100, batch_size=32, lr=1e-4
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
	    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
	    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	}



	# model
	n_class = len(np.unique(y_train))
	model = TwoLayerNet(
		input_size=X_train.shape[1],
		hidden_size=16,
		num_classes=n_class
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

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
		logits     PyTorch Tensor       (1442, 2)                   11.27 KB torch.float32
		y          PyTorch Tensor       (1442,)                     11.27 KB torch.int64
		"""

		f1_train = calculate_f1(train_data, name="train")
		f1_test = calculate_f1(test_data, name="test")

		mt.update(epoch, **train_loss, **test_loss, **f1_train, **f1_test)
		mt.print_latest()

	df = mt.generate_df()
	max_f1_dict = df.loc[df["f1_class_macro_test"].idxmax()].to_dict()
	"""
	{
		'epoch': 20.0, 
		'train_loss': 0.20217878970762956, 
		'test_loss': 0.24036466106772422, 
		'f1_class_0_train': 0.9549980163574219, 
		'f1_class_1_train': 0.697050929069519, 
		'f1_class_macro_train': 0.8260244727134705, 
		'f1_class_0_test': 0.9438877701759338, 
		'f1_class_1_test': 0.7666666507720947, 
		'f1_class_macro_test': 0.8552771806716919
	}
	"""
	return max_f1_dict



if __name__ == "__main__":

	# concatenate
	X = np.concatenate(list_eeg, axis=0)
	y = np.concatenate(list_label, axis=0)
	feature_names = [f'feature_{i}' for i in range(X.shape[1])]
	"""
	X        NumPy Array          (2061, 64)                 515.25 KB float32
	y        NumPy Array          (2061,)                     16.10 KB int64
	"""

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=42
	)

	res = score_neural_net(X_train, y_train, X_test, y_test, n_epoch=100, lr=1e-4)
