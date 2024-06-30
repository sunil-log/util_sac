import numpy as np
from collections import defaultdict

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

	def get_values(self):
		return [value for _, value in self.data]

	def get_average(self):
		values = self.get_values()
		return np.mean(values) if values else None


class metric_tracker:
	def __init__(self):
		self.metrics = {}

	def update(self, epoch, **kwargs):
		for metric_name, value in kwargs.items():
			if metric_name not in self.metrics:
				self.metrics[metric_name] = Metric(metric_name)
			self.metrics[metric_name].update(epoch, value)

	def get_latest(self, metric_name):
		if metric_name in self.metrics:
			latest = self.metrics[metric_name].get_latest()
			return latest[1] if latest else None
		return None

	def get_all(self, metric_name):
		if metric_name in self.metrics:
			return self.metrics[metric_name].get_all()
		return []

	def __getitem__(self, metric_name):
		return self.get_all(metric_name)

	def get_metrics(self):
		return list(self.metrics.keys())

	def get_average(self, metric_name):
		if metric_name in self.metrics:
			return self.metrics[metric_name].get_average()
		return None

	def print_latest(self):
		output = ", ".join(
			f"{metric_name}: {self.get_latest(metric_name) if self.get_latest(metric_name) is not None else '데이터 없음'}"
			for metric_name in self.get_metrics()
		)
		print(output, flush=True)

	def remove_metric(self, metric_name):
		if metric_name in self.metrics:
			del self.metrics[metric_name]

	def clear_data(self, metric_name=None):
		if metric_name:
			if metric_name in self.metrics:
				self.metrics[metric_name] = Metric(metric_name)
		else:
			self.metrics.clear()