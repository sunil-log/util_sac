
# -*- coding: utf-8 -*-
"""
Created on  Mar 01 2025

@author: sac
"""



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
from types import SimpleNamespace
from util_sac.pytorch.optuna.sample_params import sample_params


lr_dicts = [
    {10: 1e-4, 40: 1e-5},
    {50: 1e-4, 80: 1e-5},
    {99: 1e-4, 100: 1e-5}
]


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
	"lr_dict_idx": {
		"type": "int",
		"low": 0,
		"high": len(lr_dicts) - 1,
		"step": 1,
		"log": False
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
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(sub_dir_list, trial_name=args.trial_name, zip_src_loc="../")

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
			mt.plot_metric(axes[1, 0], keys=["lr"], y_log='log')
			plt.tight_layout()
			plt.savefig(tm.trial_dir / "train_test_loss.png")



		# save metrics
		df_metrics = mt.generate_df()
		df_metrics.to_csv(f"{tm.trial_dir}/train_test_metrics.csv", index=False)
		save_df_as_npz(df_metrics, f"{tm.trial_dir}/train_test_metrics.npz")

		# best score
		best_f1 = df_metrics["f1_class_macro_test"].max()


		# save hyperparameters
		args.best_score = best_f1
		save_args(args, f"{tm.trial_dir}/hyperparameters.json")


		return best_f1


def get_objective(study_name: str):
	def objective(trial: optuna.trial.Trial) -> float:
		# 여기서, study_name이 클로저 형태로 바깥에서 넘어온다.
		# (예: trial_name 설정 시 study_name과 trial.number를 묶어서 사용)

		args_dict = sample_params(trial, param_space)
		args = SimpleNamespace(**args_dict)
		args.lr_dict = lr_dicts[args.lr_dict_idx]

		# trial 이름을 일관성 있게 지정
		args.optuna_trial_index = trial.number
		args.trial_name = f"{study_name}__Trial_{trial.number}"  # 원하는 형식으로

		# 이제 train_session에 넘겨서 학습
		score = train_session(args)
		return score

	return objective


def main():

	"""
	main
	"""

	"""
	1. Single Trial
	"""
	args_dict = {
		"trial_name": "Building a new model",
		"input_dim": 32,
		"n_head": 8,
		"q_dim": 16,
	}
	args = SimpleNamespace(**args_dict)
	score = train_session(args)
	print("Single Trial Score:", score)
	exit()

	"""
	2. Optuna Optimization 
	"""
	# 1) study 생성 (이미 존재하면 로드)
	study_name = "my_study2"
	study = optuna.create_study(
		study_name=study_name,
		storage=f"sqlite:///{study_name}.db",
		load_if_exists=True,
		direction="maximize"
	)

	# get_objective(study_name)로부터 objective 함수를 얻어서 optimize
	objective_func = get_objective(study_name)
	study.optimize(objective_func, n_trials=100)

	print("Best value:", study.best_value)
	print("Best params:", study.best_params)



if __name__ == "__main__":
	main()
