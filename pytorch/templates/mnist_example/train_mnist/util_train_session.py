
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

@author: sac
"""


import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from util_sac.pytorch.data.trial_manager import trial_manager
from util_sac.pytorch.data.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trainer.update_lr import current_lr
from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1s
from util_sac.dict.json_manager import save_json
from util_sac.pandas.save_npz import save_df_as_npz

from train_mnist.util_dataloader import load_data
from train_mnist.util_model import TwoLayerMNIST
from train_mnist.util_trainer import NewTrainer



def plot_losses(tm, mt):


	# plot the train and test metrics
	plt.close()
	fig, axes = plt.subplots(2, 3, figsize=(20, 10))

	# plot losses
	mt.plot_metric(axes[0, 0], keys=["train_loss", "valid_loss", "test_loss"], y_log='log')
	# mt.plot_metric(axes[0, 1], keys=["train_accuracy", "test_accuracy"])
	mt.plot_metric(axes[0, 2], keys=["f1_class_macro_train", "f1_class_macro_valid", "f1_class_macro_test"])
	mt.plot_metric(axes[1, 0], keys=["lr"], y_log='log')
	plt.tight_layout()
	plt.savefig(tm.trial_dir / "train_test_loss.png")



def train_session(args):

	# 2) trial manager
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(
		sub_dir_list,
		base_dir=args["db_dir"],
		trial_name=args["trial_name"],
		zip_src_loc="../"
	)


	# load data
	dataloaders = load_data(args)
	"""
	train      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	valid      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	test       Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A  
	"""

	# 4) Model 생성
	model = TwoLayerMNIST(args)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()


	# 5) Trainer 생성
	trainer = NewTrainer(
		model=model,
		dataloaders=dataloaders,
		optimizer=optimizer,
		criterion=criterion,
		lr_dict=args["lr_dict"],
		n_epoch=args["n_epoch"],
		args=args,
	)


	"""
	Start training
	"""
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):

		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		valid_loss, valid_data = trainer.one_epoch(mode='valid', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)
		lr = current_lr(optimizer)

		losses = {**train_loss, **valid_loss, **test_loss}
		f1s = calculate_f1s(train_data, valid_data, test_data)

		mt.update(epoch, **losses, **f1s, **lr)
		mt.print_latest()


		# save model
		if args["save"]["model"] is not None:
			if epoch % args["save"]["model"] == 0:
				torch.save(model.state_dict(), f"{tm['weights']}/epoch_{epoch}.pth")

		# save plot
		if args["save"]["plot"] is not None:
			if epoch % args["save"]["plot"] == 0:
				plot_losses(tm, mt)



	"""
	After training
	"""

	# save plot
	plot_losses(tm, mt)

	# save metrics
	df_metrics = mt.generate_df()
	df_metrics.to_csv(f"{tm.trial_dir}/train_test_metrics.csv", index=False)
	save_df_as_npz(df_metrics, f"{tm.trial_dir}/train_test_metrics.npz")

	# save hyperparameters
	save_json(args, f"{tm.trial_dir}/hyperparameters.json")


	# smoothing 후 best score 계산
	# df_metrics["s1"] = df_metrics["auc_roc_valid"].rolling(window=5).mean()
	# df_metrics["s2"] = df_metrics["auc_roc_test"].rolling(window=5).mean()
	# df_metrics = df_metrics.dropna().reset_index(drop=True)
	# df_metrics["score"] = df_metrics["s1"] * df_metrics["s2"]

	# idx_best = df_metrics["score"].idxmax()  # 여기서 idx_best는 0부터 시작하는 정수
	# best_roc = df_metrics["s2"].iloc[idx_best]

	# args.best_score = best_roc

	best_roc = 0
	return best_roc