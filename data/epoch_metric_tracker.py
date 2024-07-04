

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
			if key in self.metrics:
				epochs, values = self.get_all(key)
				label = f"{key}: {values[-1]:.4f}"
				ax.plot(epochs, values, label=label)

		ax.set_xlabel('Epoch')
		ax.set_ylabel('Value')
		ax.legend()
		ax.grid(True)

		title = f"Epoch: {epochs[-1]}; {keys[0]}"
		ax.set_title(title)

		if y_log:
			ax.set_yscale('log')



if __name__ == '__main__':

	# train
	n_epoch = 10
	mt = metric_tracker()
	for epoch in range(n_epoch):

		train_loss = trainer.one_epoch(if_train=True)
		test_loss = trainer.one_epoch(if_train=False)

		mt.update(epoch, **train_loss, **test_loss)
		mt.print_latest()

