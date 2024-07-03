

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
		for metric_name, value in kwargs.items():
			if metric_name not in self.metrics:
				self.metrics[metric_name] = Metric(metric_name)
			self.metrics[metric_name].update(epoch, value)

	def get_latest(self, metric_name):
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

