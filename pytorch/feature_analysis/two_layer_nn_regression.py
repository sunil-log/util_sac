
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset

from util_sac.pytorch.data import metric_tracker
from util_sac.pytorch.trainer.trainer import BaseTrainer


class FeatureAttributionInspector:
	def __init__(self, model, X_all, y_all, feature_names):
		"""
		model: 이미 학습된 PyTorch 모델
		X_all: 전체 (train+test) Feature (numpy.ndarray, shape: [N, D])
		y_all: 전체 (train+test) Target (numpy.ndarray, shape: [N])
		feature_names: Feature 이름 리스트 (ex: ['input_dim', 'n_head', 'q_dim'])
		"""
		self.model = model
		self.X_all = X_all
		self.y_all = y_all
		self.feature_names = feature_names

		# 모델의 디바이스 추론 후 맞춰두기
		self.device = next(model.parameters()).device
		self.model.eval().to(self.device)

		# Captum IntegratedGradients 객체
		self.ig = IntegratedGradients(self.model)

	def top_k_indices(self, k=3):
		"""
		y_all에서 값이 큰 상위 k개 인덱스를 반환한다
		"""
		# numpy.ndarray이면 argsort() 활용
		# (-y_all)을 정렬하면 큰 값부터 인덱스 확보 가능
		idx_sorted = self.y_all.argsort()[::-1]  # 내림차순
		idx_top_k = idx_sorted[:k]
		return idx_top_k

	def interpret_one_sample(self, x, baseline=None, n_steps=50):
		"""
		X 하나의 샘플 x (shape: [D])에 대하여 IntegratedGradients로 Feature 기여도를 계산
		baseline: None이면 0 벡터 사용
		n_steps: alpha 보간 단계(기본 50)
		return: (attributions, delta)
		"""
		# x -> (1, D) 텐서
		x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)

		if baseline is None:
			baseline_tensor = torch.zeros_like(x_tensor)
		else:
			baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0).to(self.device)

		# 회귀 문제이므로 target=None or target=0 지정 가능
		attributions, delta = self.ig.attribute(
			x_tensor,
			baselines=baseline_tensor,
			n_steps=n_steps,
			return_convergence_delta=True
		)

		return attributions, delta

	def interpret_top_k(self, k=3, baseline=None, n_steps=50):
		"""
		y가 큰 상위 k개에 대해서 IntegratedGradients 결과를 출력
		"""
		# 상위 k개 인덱스 추출
		idx_top_k = self.top_k_indices(k)

		for rank, idx in enumerate(idx_top_k, start=1):
			x_sample = self.X_all[idx]
			y_value = self.y_all[idx]

			# IG 계산
			attributions, delta = self.interpret_one_sample(
				x_sample,
				baseline=baseline,
				n_steps=n_steps
			)

			# CPU로 가져와서 numpy 변환
			attributions_np = attributions.detach().cpu().numpy()[0]  # shape: (D,)
			delta_np = float(delta.detach().cpu().numpy())

			# 출력
			print(f"--- Top {rank} ---")
			print(f"Sample index: {idx}")
			print(f"X: {x_sample} -> y: {y_value:.4f}")
			for f_name, attr_value in zip(self.feature_names, attributions_np):
				print(f"  Feature: {f_name:>10} | Attribution: {attr_value:.4f}")
			print(f"Convergence Delta: {delta_np:.6f}")
			print("", flush=True)


class TwoLayerNetTrainer(BaseTrainer):

	def one_step(self, batch, epoch):
		X, y = batch
		"""
		X		  PyTorch Tensor	   (32, 64)					 8.00 KB torch.float32
		y		  PyTorch Tensor	   (32,)					   256.00 B torch.int64
		"""

		X = X.to(self.device)
		y = y.to(self.device)

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

	model, df = train_nn_regression(
		X_train, y_train, X_test, y_test, n_epoch=100, lr=1e-3
	)

	# 인스펙터 생성
	inspector = FeatureAttributionInspector(
		model=model,
		X_all=X,
		y_all=y,
		feature_names=feature_names
	)

	# 상위 3개 해석
	inspector.interpret_top_k(k=3)

	"""
	각 Attribution 은 local linear regression 의 coefficient 비스무리 한 의미를 가진다.
	
	--- Top 1 ---
	Sample index: 106
	X: [ 8 64 32] -> y: 0.8581
	  Feature:  input_dim | Attribution: -0.3412
	  Feature:     n_head | Attribution: 1.3516
	  Feature:      q_dim | Attribution: 0.0779
	Convergence Delta: 0.009939
	
	--- Top 2 ---
	Sample index: 176
	X: [32 64  8] -> y: 0.8498
	  Feature:  input_dim | Attribution: -1.3414
	  Feature:     n_head | Attribution: 1.3337
	  Feature:      q_dim | Attribution: 0.0111
	Convergence Delta: 0.005541
	
	--- Top 3 ---
	Sample index: 177
	X: [32 64 16] -> y: 0.8481
	  Feature:  input_dim | Attribution: -1.3559
	  Feature:     n_head | Attribution: 1.3526
	  Feature:      q_dim | Attribution: 0.0294
	Convergence Delta: 0.009286
	"""