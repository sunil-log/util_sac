import torch
from util_sac.data.batch_data_collector import batch_loss_collector

"""
one_step 만 implement 하면 된다.
	1. batch 를 받아서 parameter 를 정리하고,
	2. forward pass 를 수행하여 loss 를 계산하고,
	3. total_loss 는 return 하고, metric 을 self.loss_tracker 에 update 한다.
		- update 시에 "train_" prefix 는 붙일 필요 없다
		- update 하는 loss 는 .item() 을 호출하여 scalar 로 변환하여 넣어야 한다.
	
	
class VAETrainer(BaseTrainer):
	def one_step(self, batch):
		# data
		x, _ = batch
		x = x.to(self.device)

		# forward
		x_hat, mu, log_var = self.model(x)
		loss_r, loss_kl = self.criterion(x, x_hat, mu, log_var)
		loss = loss_r + loss_kl

		# collect loss
		self.loss_tracker.update(
			reconstruction_loss=loss_r.item(),
			kl_divergence=loss_kl.item(),
			total_loss=loss.item()
		)
		return loss
"""


class BaseTrainer:
	def __init__(self, model, train_loader, test_loader, optimizer, criterion):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(self.device)

		# one_step 에서 사용해야 하므로 self 로 선언
		self.loss_collector = None

	def one_epoch(self, if_train=True):
		# set model mode
		if if_train:
			self.model.train()
			data_loader = self.train_loader
		else:
			self.model.eval()
			data_loader = self.test_loader

		# initialize loss tracker
		self.loss_collector = batch_loss_collector()

		# iterate over data
		for batch_idx, batch in enumerate(data_loader):
			# forward
			self.optimizer.zero_grad()
			loss = self.one_step(batch)
			# backward
			if if_train:
				loss.backward()
				self.optimizer.step()

		# return list of loss
		d_loss = self.loss_collector.average()

		# modify keys and return
		if if_train:
			d_loss = {"train_" + k: v for k, v in d_loss.items()}
		else:
			d_loss = {"test_" + k: v for k, v in d_loss.items()}
		return d_loss

	def one_step(self, batch):
		raise NotImplementedError("Subclasses should implement this!")


