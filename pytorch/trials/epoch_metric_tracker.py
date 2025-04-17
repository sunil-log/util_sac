

"""
metric_tracker 모듈
==================

본 모듈은 **딥러닝 학습 과정에서 발생하는 다양한 scalar metric**(예: `loss`, `accuracy`,
`AUC`)을 **epoch 단위로 기록‧조회‧시각화**하기 위해 설계된 경량 추적기(tracker)이다.
`torch.Tensor` ‑‑> `float` 자동 변환, `numpy.ndarray` 단일 원소 지원,
`matplotlib` 축(ax) 전달 기반의 다중 곡선 플로팅 등 **실험 관리에서 자주 반복되는
보일러플레이트 코드를 제거**하는 데 목적이 있다.

주요 구성
--------
* **Metric**
  개별 metric 하나를 표현하는 경량 컨테이너. `(epoch, value)` 튜플 시퀀스를 보존한다.
* **metric_tracker**
  여러 `Metric` 인스턴스를 **dict 형태로 관리**하며,
  - `update(epoch, **kwargs)` : 한 번의 호출로 여러 metric 갱신
  - `get_latest(name)` : 가장 최근 (epoch, value)
  - `__getitem__(name)` : `(epochs, values)` 언패킹용 sugar syntax
  - `plot_metric(ax, keys, …)` : 복수 metric 곡선 시각화
  - `generate_df()` : `pandas.DataFrame` 로 직렬화

의존 / 호환
-----------
* **필수** : `numpy`, `pandas`, `torch`
* **선택** : `matplotlib` (시각화 기능 사용 시)
* Python ≥ 3.8, PyTorch ≥ 1.10 권장

사용 예시
---------
```python
import matplotlib.pyplot as plt
from metric_tracker import metric_tracker

mt = metric_tracker()
for epoch in range(num_epochs):
    train_stats = trainer.one_epoch(train=True)   # e.g. {'train_loss': 0.123}
    valid_stats = trainer.one_epoch(train=False)  # e.g. {'valid_loss': 0.145}

    mt.update(epoch, **train_stats, **valid_stats)
    mt.print_latest()          # 콘솔에 'Epoch 7 | train_loss: 0.1234 | …' 형식 출력

# 시각화
fig, ax = plt.subplots()
mt.plot_metric(ax, keys=["train_loss", "valid_loss"], y_log=True)
plt.show()

# 로그를 DataFrame 으로 저장
df = mt.generate_df()
df.to_csv("training_log.csv", index=False)
"""


import numpy as np
import pandas as pd
import torch


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


	def plot_metric(self, ax, keys, y_log=False, y_range=None):
		"""
		Plot multiple metrics on the given axes.

		Args:
			ax (matplotlib.axes.Axes): The axes to plot on.
			keys (list): List of metric names to plot.
			y_log (bool): Whether to use log scale for y-axis. Default is False.
			y_range (tuple or list, optional): y-axis 범위를 지정한다. 예: (0, 1).


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

		# y-axis 범위 설정
		if y_range is not None:
			ax.set_ylim(y_range)

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

