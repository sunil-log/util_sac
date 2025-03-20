
# -*- coding: utf-8 -*-
"""
Created on  Mar 01 2025

@author: sac
"""


from types import SimpleNamespace

import torch
import matplotlib.pyplot as plt

from util_sac.data.trial_manager2 import trial_manager
from util_sac.data.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trainer.trainer import BaseTrainer
from util_sac.pytorch.trainer.update_lr import current_lr
from util_sac.pytorch.load_data.move_device import move_dict_tensors_to_device
from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1
from util_sac.dict.save_args import save_args
from util_sac.pandas.save_npz import save_df_as_npz

import optuna
from util_sac.pytorch.optuna.sample_params import sample_params


param_space = {
	"input_dim": {
		"type": "categorical",
		"choices": [2, 4, 8, 16, 32, 64, 128]
	},
	"n_head": {
		"type": "categorical",
		"choices": [2, 4, 8, 16, 32, 64, 128]
	},
	"q_dim": {
		"type": "categorical",
		"choices": [2, 4, 8, 16, 32, 64, 128]
	},
	"lr_dict": {
		"type": "categorical",
		"choices": [
			{10: 1e-4, 40: 1e-5},
			{50: 1e-4, 80: 1e-5},
			{99: 1e-4, 100: 1e-5},
		]
	}
}



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



def train_session(args):


	# 2) trial manager
	trial_name = f"ID_2151__ts2vec"
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(sub_dir_list, trial_name=trial_name, zip_src_loc="../../")


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


	# 5) Trainer 생성
	trainer = NewTrainer(
		model=model,
		dataloaders=dataloaders,
		optimizer=optimizer,
		criterion=criterion,
		lr_dict=args.lr_dict,
		n_epoch=100,
		args=args,
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
		save_df_as_npz(df_metrics, f"{tm.trial_dir}/train_test_metrics.npz")
		save_args(args, f"{tm.trial_dir}/hyperparameters.json")


		# best score
		best_f1 = df_metrics["f1_class_macro_test"].max()
		return best_f1


def objective(trial: optuna.trial.Trial):
	# 2) sample_params 함수를 통해 dict 획득
	args_dict = sample_params(trial, param_space)

	# 3) SimpleNamespace로 감싸서 사용 (편의를 위해)
	args = SimpleNamespace(**args_dict)

	# 4) 이 args를 활용해 학습/검증
	score = train_session(args)

	return score


def main():

	"""
	main
	"""

	"""
	1. Single Trial
	"""
	args_dict = {
		"input_dim": 32,
		"n_head": 8,
		"q_dim": 16
	}
	args = SimpleNamespace(**args_dict)
	score = train_session(args)
	print("Single Trial Score:", score)
	exit()

	"""
	2. Optuna Optimization 
	"""
	# 이미 study가 존재하면 불러오고, 없다면 새로 만든다
	study = optuna.create_study(
		study_name="my_study",
		storage="sqlite:///my_study.db",
		load_if_exists=True,
		direction="maximize"
	)

	# 원하는 만큼 trial 실행
	study.optimize(objective, n_trials=2)

	print("Best value:", study.best_value)
	print("Best params:", study.best_params)



if __name__ == "__main__":
	main()
