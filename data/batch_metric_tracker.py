import torch
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

class batch_loss_tracker:
	def __init__(self):
		self.losses = defaultdict(float)
		self.num_batches = 0

	def update(self, loss_dict):
		for key, value in loss_dict.items():
			self.losses[key] += value.item()
		self.num_batches += 1

	def average(self):
		"""
		loss 가 mini-batch 안에서 sample 에 대한 평균이기 때문에,
		각 loss 에 대해서 num_batches 로 나눠주어야 한다.

		return {'train_loss': 0.5278644561767578, 'val_loss': 0.24387567241986594}
		"""
		return {key: value / self.num_batches for key, value in self.losses.items()}

	def reset(self):
		self.losses.clear()
		self.num_batches = 0