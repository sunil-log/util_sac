import torch
from pathlib import Path
import pandas as pd
from util_sac.pytorch.data import batch_loss_collector, batch_data_collector
from util_sac.pytorch.trainer.update_lr import update_lr_with_dict

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
	
    def __init__(self, **kwargs):
        # BaseTrainer를 상속받으며, my_param 파라미터를 추가로 받는 클래스입니다.
        super().__init__(**kwargs)
        
        # 여기서 검증이나 로직이 필요하면 명시적으로 꺼낸다.
        if not hasattr(self, 'my_param'):
            raise ValueError("my_param이 필요합니다.")
        
        if self.my_param < 0:
            print("my_param이 음수입니다. 0으로 보정합니다.")
            self.my_param = 0

        # 필요한 추가 초기화 ...


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


from torch.optim.lr_scheduler import LambdaLR

lr_schedule = {
	0: 1.0,	   # 0-49 에폭: 초기 학습률 유지 (초기 학습률: 0.01)
	100: 0.1,   # 50-299 에폭: 초기 학습률의 1/10
	800: 0.01  # 300 에폭 이후: 초기 학습률의 1/100
}

def lr_lambda(epoch):
	return next((mult for start, mult in sorted(lr_schedule.items(), reverse=True) if epoch >= start), 1.0)


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
	lr_schedule = {
		10: 1e-4,
		100: 1e-5,
	}
	
	trainer = VAETrainer(			
				model=vae,
				dataloaders=dataloaders,
				optimizer=optimizer,
				criterion=vae_loss,
				n_epoch=100,
				lr_dict=lr_schedule,
				other_param="hello")


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
		
		print_array_info(train_data)

		
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
			plt.tight_layout()
			plt.savefig(tm.trial_dir / "train_test_loss.png")
			
	
	# read all train_test_metrics.csv files
	base_path = Path('./trials')  # 현재 작업 디렉토리
	keywords = ['classify_stage', '__epoch_1000']
	result_df = concat_train_test_metrics(keywords)
	print(result_df)
"""



class BaseTrainer:
	def __init__(
			self,
			model=None,
			dataloaders=None,
			optimizer=None,
			criterion=None,
			n_epoch=None,
			lr_dict=None,
			**kwargs
	):

		"""
		:param model: PyTorch model
		:param dataloaders: dict 형태의 dataloaders (예: {"train": ..., "val": ...})
		:param optimizer: PyTorch optimizer
		:param criterion: loss function
		:param n_epoch: 학습 epoch 수
		:param kwargs: 추가로 받아야 할 파라미터가 있을 경우를 대비 (확장성)
		"""
		# 필수 파라미터 체크 (예: None이면 에러)
		if model is None:
			raise ValueError("model is required")
		if dataloaders is None:
			raise ValueError("dataloaders is required")
		if optimizer is None:
			raise ValueError("optimizer is required")
		if criterion is None:
			raise ValueError("criterion is required")
		if n_epoch is None:
			raise ValueError("n_epoch is required")


		self.model = model
		self.dataloaders = dataloaders
		self.optimizer = optimizer
		self.criterion = criterion
		self.lr_dict = lr_dict

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

		# kwargs로 추가 전달된 파라미터들 저장 (필요한 경우)
		for k, v in kwargs.items():
			setattr(self, k, v)


	def one_epoch(self, mode, epoch):

		# update lr
		if mode == 'train' and self.lr_dict:
			update_lr_with_dict(self.optimizer, epoch, self.lr_dict)

		# set mode
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
					self.one_step(batch, epoch)

		# return list of loss
		d_loss = self.loss_collector.average()
		d_data = self.data_collector.get_collected_data()

		# modify keys and return
		d_loss = {f"{mode}_{k}": v for k, v in d_loss.items()}
		return d_loss, d_data

	def one_step(self, batch, epoch):
		raise NotImplementedError("Subclasses should implement this!")