import torch
from util_sac.data.batch_data_collector import batch_loss_collector, batch_data_collector

"""
trainer.py 대비 dataloader 를 dict 로 모아서 받는다.

one_step 만 implement 하면 된다.
	1. batch 를 받아서 parameter 를 정리하고,
	2. forward pass 를 수행하여 loss 를 계산하고,
	3. total_loss 는 return 하고, metric 을 self.loss_tracker 에 update 한다.
		- update 시에 "train_" prefix 는 붙일 필요 없다
		- update 하는 loss 는 .item() 을 호출하여 scalar 로 변환하여 넣어야 한다.

Example:
	- [[concept, VAE, part 03, Implementation - MNIST (beta-VAE)]]

class VAETrainer(BaseTrainer):

	def one_step(self, batch, epoch):
	
		# Progress: epoch/self.n_epoch
	
		# data
		x, label = batch
		x *= global_scale_factor
		x = x.to(self.device)

		# forward
		x_mu, z_mu, z_log_var, z_sample = self.model(x)
		kl, uncertainty, reconstruction = self.criterion(z_mu, z_log_var, x, x_mu, self.model.x_log_var)
		loss = uncertainty + reconstruction + global_beta * kl


		# collect loss
		self.loss_collector.update(
			reconstruction_loss=reconstruction.item(),
			uncertainty=uncertainty.item(),
			kl_divergence=kl.item(),
			total_loss=loss.item(),
			x_log_var=self.model.x_log_var.item()
		)

		# collect test data
		if self.mode == 'test':
			self.data_collector.update(
				x=x,
				x_hat=x_mu,
				z=z_sample,
				label=label
			)

		return loss


def main():

	# dataloader
	dataloaders = {
	    'train': train_loader,
	    'valid': valid_loader,
	    'test': test_loader
	}

	# build model
	vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=10)
	optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
	trainer = VAETrainer(vae, dataloaders, optimizer, vae_loss, n_epoch=100)


	# train
	from util_sac.data.epoch_metric_tracker import metric_tracker
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):

		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		valid_loss, valid_data = trainer.one_epoch(mode='valid', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		mt.update(epoch, **train_loss, **test_loss)
		mt.print_latest()

		plt.hist(test_data['x_hat'].flatten(), bins=30, density=True)
		
		if epoch % 5 == 0:

			# save model
			torch.save(model.state_dict(), f"{tm['weights']}/epoch_{epoch}.pth")

			# save metrics
			df_metrics = mt.generate_df()
			df_metrics.to_csv(f"{tm.trial_dir}/train_test_metrics.csv", index=False)

			# plot the train and test metrics
			plt.close()
			fig, axes = plt.subplots(2, 3, figsize=(20, 10))

			# plot losses
			mt.plot_metric(axes[0, 0], keys=["train_loss", "valid_loss", "test_loss"], y_log='log')
			mt.plot_metric(axes[0, 1], keys=["valid_accuracy", "test_accuracy"])
			mt.plot_metric(axes[0, 2], keys=["valid_macro_f1", "test_macro_f1"])
"""



class BaseTrainer:
	def __init__(self, model, dataloaders, optimizer, criterion, n_epoch):
		self.model = model
		self.dataloaders = dataloaders
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(self.device)

		# one_step 에서 사용해야 하므로 self 로 선언
		self.loss_collector = None
		self.data_collectors = None

		# n_epoch
		self.n_epoch = n_epoch + 1
		if not isinstance(self.n_epoch, int):
			raise ValueError("n_epoch should be integer type")

		# mode
		self.mode = "train"

	def one_epoch(self, mode, epoch):
		self.mode = mode
		self.model.train() if mode == 'train' else self.model.eval()
		data_loader = self.dataloaders[mode]

		# initialize loss tracker
		self.loss_collector = batch_loss_collector()
		self.data_collector = batch_data_collector()

		# iterate over data
		for batch_idx, batch in enumerate(data_loader):
			self.optimizer.zero_grad()
			if mode == 'train':
				try:
					loss = self.one_step(batch, epoch)
					if loss == "skip_update":
						continue
					if loss is None:
						raise ValueError("loss should be returned")
					loss.backward()
					self.optimizer.step()
				except Exception as e:
					print(f"Error in batch {batch_idx}: {e}")
			else:
				# only forward - no loss returned
				with torch.no_grad():
					self.one_step(batch)

		# return list of loss
		d_loss = self.loss_collector.average()
		d_data = self.data_collector.get_collected_data()

		# modify keys and return
		d_loss = {f"{mode}_{k}": v for k, v in d_loss.items()}
		return d_loss, d_data

	def one_step(self, batch, epoch):
		raise NotImplementedError("Subclasses should implement this!")