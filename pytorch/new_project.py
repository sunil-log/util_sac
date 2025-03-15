
# -*- coding: utf-8 -*-
"""
Created on  Mar 01 2025

@author: sac
"""


import argparse

import torch

import matplotlib.pyplot as plt

from util_sac.data.trial_manager2 import trial_manager
from util_sac.data.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trainer.trainer import BaseTrainer
from util_sac.pytorch.trainer.update_lr import current_lr
from util_sac.pytorch.load_data.move_device import move_dict_tensors_to_device
from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1
from util_sac.dict.save_args import save_args




class NewTrainer(BaseTrainer):

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

		# data
		d = {
			"x": x,
			"label": label
		}
		d = move_dict_tensors_to_device(d, self.device)
		# print_array_info(data)

		# forward
		y_hat = self.model(d['x'])
		loss = self.criterion(y_hat, d['label'])

		# collect loss
		self.loss_collector.update(
			loss=loss.item(),
		)

		# collect test data
		if self.mode == 'test':
			self.data_collector.update(
				logits=y_hat,  # logits 가 있어야 metric 계산 가능
				y=batch['label'],  # y 가 있어야 metric 계산 가능
			)

		return loss


def parse_arguments():

	parser = argparse.ArgumentParser(description="Hyperparameters")

	parser.add_argument(
		"--signal_type",
		type=str,
		choices=["EEG", "EOG", "EMG"],
		required=True,
		help="EEG, EOG, EMG 중 하나를 선택하세요."
	)

	args = parser.parse_args()

	return args



def main():

	# 1) Hyperparameters with argparse
	args = parse_arguments()
	signal_type = args.signal_type


	# 2) trial manager
	trial_name = f"ID_2151__ts2vec_{signal_type}"
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(sub_dir_list, trial_name=trial_name, zip_src_loc="../")


	# 3) load data
	dataloader_train = None
	dataloader_valid = None
	dataloader_test = None
	dataloaders = {
		'train': dataloader_train,
		'valid': dataloader_valid,
		'test': dataloader_test
	}

	# 4) Model 생성
	model = None
	optimizer = None
	criterion = None
	lr_schedule = {
		10: 1e-4,
		100: 1e-5,
	}


	# 5) Trainer 생성
	trainer = NewTrainer(
		model=model,
		dataloaders=dataloaders,
		optimizer=optimizer,
		criterion=criterion,
		lr_dict=lr_schedule,
		n_epoch=100,
	)


	# 6) train
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):

		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		valid_loss, valid_data = trainer.one_epoch(mode='valid', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)
		lr = current_lr(optimizer)

		f1_train = calculate_f1(train_data, name="train")
		f1_valid = calculate_f1(valid_data, name="valid")
		f1_test = calculate_f1(test_data, name="test")

		mt.update(epoch, **train_loss, **test_loss, **f1_train, **f1_test, **lr)
		mt.print_latest()

		if epoch % 10 == 0:

			# save model
			torch.save(model.state_dict(), f"{tm['weights']}/epoch_{epoch}.pth")

			# plot the train and test metrics
			plt.close()
			fig, axes = plt.subplots(2, 3, figsize=(20, 10))

			# plot losses
			mt.plot_metric(axes[0, 0], keys=["train_loss", "valid_loss", "test_loss"], y_log='log')
			mt.plot_metric(axes[0, 1], keys=["train_accuracy", "test_accuracy"])
			mt.plot_metric(axes[0, 2], keys=["f1_class_macro_train", "f1_class_macro_test"])
			plt.tight_layout()
			plt.savefig(tm.trial_dir / "train_test_loss.png")


		# save metrics
		df_metrics = mt.generate_df()
		df_metrics.to_csv(f"{tm.trial_dir}/train_test_metrics.csv", index=False)
		save_args(args, f"{tm.trial_dir}/hyperparameters.json")





if __name__ == "__main__":
	main()
