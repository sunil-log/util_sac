
# -*- coding: utf-8 -*-
"""
Created on  Mar 01 2025

@author: sac
"""



from types import SimpleNamespace

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from util_sac.dict.json_manager import save_json
from util_sac.pandas.save_npz import save_df_as_npz
from util_sac.pytorch.trials.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trials.trial_manager import trial_manager
from util_sac.pytorch.dataloader.to_tensor_device import move_dict_tensors_to_device
from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1
from util_sac.pytorch.optuna.get_objective import generate_lr_schedules
from util_sac.pytorch.trainer.trainer import BaseTrainer
from util_sac.pytorch.trainer.update_lr import current_lr

n_epoch = 15
n_trials = 100
lr_dicts = generate_lr_schedules(
	num_schedules=50,
	total_epochs=n_epoch
)



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

		# trials
		d = {
			"x": None,
			"label": None
		}
		d = move_dict_tensors_to_device(d, self.device)
		# print_array_info(trials)

		# forward
		y_hat = self.model(d['x'])
		loss = self.criterion(y_hat, d['label'])

		# collect loss
		self.loss_collector.update(
			loss=loss.item(),
		)

		# collect test trials
		if self.mode == 'test':
			self.data_collector.update(
				logits=y_hat,  # logits 가 있어야 metric 계산 가능
				y=batch['label'],  # y 가 있어야 metric 계산 가능
			)

		return loss



def train_session(args):

	# 2) trial manager
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(sub_dir_list, trial_name=args.trial_name, zip_src_loc="../../")

	# 3) load trials
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
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()


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


	# smoothing 후 best score 계산
	df_metrics["s1"] = df_metrics["auc_roc_valid"].rolling(window=5).mean()
	df_metrics["s2"] = df_metrics["auc_roc_test"].rolling(window=5).mean()
	df_metrics = df_metrics.dropna().reset_index(drop=True)
	df_metrics["score"] = df_metrics["s1"] * df_metrics["s2"]

	idx_best = df_metrics["score"].idxmax()  # 여기서 idx_best는 0부터 시작하는 정수
	best_roc = df_metrics["s2"].iloc[idx_best]

	args.best_score = best_roc
	save_json(args, f"{tm.trial_dir}/hyperparameters.json")


	return best_roc




def main():

	"""
	main
	"""

	args_dict = {
		"trial_name": "Building",
		"input_dim": 32,
		"n_head": 8,
		"q_dim": 16,
		"lr_dict_idx": 0,
		"fold": 0,
		"n_folds": 5,
	}
	args = SimpleNamespace(**args_dict)
	score = train_session(args)
	print("Single Trial Score:", score)




if __name__ == "__main__":
	main()
