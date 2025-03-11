
import pandas as pd
import torch
import numpy as np


class Metric:
	def __init__(self, name):
		self.name = name
		self.data = []  # (epoch, value) 튜플을 저장할 리스트

	def update(self, epoch, value):
		self.data.append((epoch, value))

	def get_latest(self):
		return self.data[-1] if self.data else None

	def get_all(self):
		return self.data

	def get_epochs_and_values(self):
		return list(zip(*self.data))

	def get_values(self):
		return [value for _, value in self.data]



class metric_tracker:
	def __init__(self):
		self.metrics = {}

	def update(self, epoch, **kwargs):
		"""
		각 metric의 값을 업데이트 할 때, 값이 1원소짜리 Tensor 혹은 numpy array이면
		이를 float으로 변환하여 저장합니다.

		:param epoch: 현재 epoch
		:param kwargs: metric 이름과 그에 대응하는 값들
		"""

		for metric_name, value in kwargs.items():

			# value를 float으로 변환
			if isinstance(value, torch.Tensor):
				value = value.item()
			elif isinstance(value, np.ndarray):
				if value.size == 1:
					value = value.item()
				else:
					raise ValueError(
						f"Expected single element numpy array for metric '{metric_name}', got array with shape {value.shape}."
					)
			else:
				value = float(value)

			# metric 업데이트
			if metric_name not in self.metrics:
				self.metrics[metric_name] = Metric(metric_name)
			self.metrics[metric_name].update(epoch, value)

	def get_latest(self, metric_name):
		"""
		:returns (epoch, value) tuple
		"""
		if metric_name in self.metrics:
			latest = self.metrics[metric_name].get_latest()
			return latest if latest else None
		return None

	def get_all(self, metric_name):
		"""
		Example:
			x, y = mt["train_loss"]
			ax.plot(*mt["train_loss"])
		"""
		if metric_name in self.metrics:
			return self.metrics[metric_name].get_epochs_and_values()
		return []

	def __getitem__(self, metric_name):
		"""
		Example:
			x, y = mt["train_loss"]
			ax.plot(*mt["train_loss"])
		"""
		return self.get_all(metric_name)

	def get_metrics(self):
		return list(self.metrics.keys())

	def print_latest(self):
		metric_names = self.get_metrics()
		list_value = []
		for metric_name in metric_names:
			latest = self.get_latest(metric_name)
			if latest is not None:
				list_value.append(f"{metric_name}: {latest[1]:.4f}")
			else:
				list_value.append(f"{metric_name}: None")
		curretn_epoch = latest[0]
		print_str = f"Epoch {curretn_epoch} | " + " | ".join(list_value)
		print(print_str, flush=True)


	def plot_metric(self, ax, keys, y_log=False):
		"""
		Plot multiple metrics on the given axes.

		Args:
			ax (matplotlib.axes.Axes): The axes to plot on.
			keys (list): List of metric names to plot.
			y_log (bool): Whether to use log scale for y-axis. Default is False.

		Example:
			fig, ax = plt.subplots()
			mt.plot_metric(ax, keys=["train_loss", "val_loss"], y_log=True)
			plt.show()
		"""
		for key in keys:
			# 해당 metric이 등록되어 있는지 확인한다.
			if key not in self.metrics:
				print(f"[Warning] '{key}'는 등록된 metric이 아니므로 스킵한다.")
				continue

			epochs, values = self.get_all(key)
			# 실제로 그릴 값이 존재하는지 확인한다.
			if not epochs or not values:
				print(f"[Warning] '{key}'에 대한 데이터가 없으므로 스킵한다.")
				continue

			label = f"{key}: {values[-1]:.4f}"
			ax.plot(epochs, values, label=label)

		ax.set_xlabel('Epoch')
		ax.set_ylabel('Value')
		ax.legend()
		ax.grid(True)

		# keys 중 첫 번째 metric의 최근 epoch 정보를 활용해 타이틀을 설정한다.
		if keys and keys[0] in self.metrics:
			epochs, _ = self.get_all(keys[0])
			if epochs:
				ax.set_title(f"Epoch: {epochs[-1]}; {keys[0]}")

		if y_log:
			ax.set_yscale('log')

	def generate_df(self):
		metrics = self.get_metrics()
		d = {'epoch': self.get_all(metrics[0])[0]}  # epoch을 먼저 추가
		for metric in metrics:
			_, values = self.get_all(metric)
			d[metric] = values
		df = pd.DataFrame(d)
		return df




if __name__ == '__main__':

	# train
	n_epoch = 10
	mt = metric_tracker()
	for epoch in range(n_epoch):

		train_loss = trainer.one_epoch(if_train=True)
		test_loss = trainer.one_epoch(if_train=False)

		mt.update(epoch, **train_loss, **test_loss)
		mt.print_latest()

