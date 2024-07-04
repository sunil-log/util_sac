import torch
from util_sac.data.batch_metric_tracker import batch_loss_tracker


class BaseTrainer:
	def __init__(self, model, train_loader, test_loader, optimizer, criterion):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(self.device)
		self.loss_tracker = None

	def one_epoch(self, if_train=True):
		# set model mode
		if if_train:
			self.model.train()
			data_loader = self.train_loader
		else:
			self.model.eval()
			data_loader = self.test_loader

		# set loss tracker
		self.loss_tracker = batch_loss_tracker()

		# iterate over data
		for batch_idx, batch in enumerate(data_loader):
			# forward
			loss = self.one_step(batch)
			# backward
			if if_train:
				loss.backward()
				self.optimizer.step()

		# return list of loss
		d_loss = self.loss_tracker.average()

		# modify keys and return
		if if_train:
			d_loss = {"train_" + k: v for k, v in d_loss.items()}
		else:
			d_loss = {"test_" + k: v for k, v in d_loss.items()}
		return d_loss

	def one_step(self, batch):
		raise NotImplementedError("Subclasses should implement this!")


