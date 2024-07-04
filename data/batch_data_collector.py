from collections import defaultdict
import numpy as np
import torch

"""
mini-batch loop 직전에 선언하여 initialize 하고,
각 mini-batch 마다 계산된 loss 를 update 하면 다음을 수 행한다.
	
	total_loss += loss.item()
	num_batches += 1

mini-batch loop 이 끝나면, .average() 를 호출하여 각 loss 에 대해 num_batches 로 나눠주어야 한다.

- 이것은 메커니즘은 update 시에 받는 loss 가,
	mini-batch 안에서 sample 에 대한 평균이라는 것을 assume 한다.
"""


class batch_loss_collector:
	def __init__(self):
		self.losses = defaultdict(float)
		self.num_batches = 0

	def update(self, **loss_dict):
		for key, value in loss_dict.items():
			self.losses[key] += float(value)
		self.num_batches += 1

	def average(self):
		"""
		loss 가 mini-batch 안에서 sample 에 대한 평균이기 때문에,
		각 loss 에 대해서 num_batches 로 나눠주어야 한다.

		return {'train_loss': 0.5278644561767578, 'val_loss': 0.24387567241986594}
		"""
		return {key: value / self.num_batches for key, value in self.losses.items()}



class batch_data_collector:

	"""
	BatchCollector는 미니배치 학습 과정에서 다양한 데이터를 수집하고 관리하는 클래스입니다.

	이 클래스는 다음과 같은 기능을 제공합니다:
	1. 다양한 형태의 데이터(예: PyTorch 텐서, NumPy 배열, 리스트)를 수집합니다.
	2. 수집된 모든 데이터를 NumPy 배열 형태로 변환하여 저장합니다.
	3. 수집이 완료된 후 모든 데이터를 하나의 NumPy 배열로 병합하여 반환합니다.
	"""

	def __init__(self):
		self.data = {}
		self.num_batches = 0

	def update(self, **batch_data):
		for key, value in batch_data.items():
			if key not in self.data:
				self.data[key] = []

			# Convert to numpy array if it's a torch tensor
			if isinstance(value, torch.Tensor):
				value = value.detach().cpu().numpy()

			# Ensure the value is a numpy array
			if not isinstance(value, np.ndarray):
				value = np.array(value)

			self.data[key].append(value)

		self.num_batches += 1

	def get_collected_data(self):
		"""
		Returns a dictionary where each value is a numpy array
		containing all collected data for that key.
		"""
		return {key: np.concatenate(value) for key, value in self.data.items()}





if __name__ == '__main__':

	loss_col = batch_loss_collector()
	data_col = batch_data_collector()

	for batch in range(10):
		loss_col.update(
			train_loss=torch.tensor(0.5),
			val_loss=torch.tensor(0.3)
		)
		data_col.update(
			x_hat = torch.randn(64, 784),
			y_hat = torch.randn(64, 10),
		)


	average_losses = loss_col.average()
	print("Average losses:", average_losses)


