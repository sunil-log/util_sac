
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

@author: sac
"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from train_iat.util_dataloader import load_data
from train_iat.util_model import IAT_model
from train_iat.util_trainer import NewTrainer


from util_sac.dict.json_manager import save_json
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pandas.save_npz import save_df_as_npz
from util_sac.pytorch.trials.epoch_metric_tracker import metric_tracker
from util_sac.pytorch.trials.trial_manager import trial_manager
from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1s
from util_sac.pytorch.metrics.binary_ROC_PR import calculate_roc_aucs
from util_sac.pytorch.trainer.update_lr import current_lr

from util_sac.pytorch.trials.best_score import compute_weighted_score


def plot_losses(tm, mt, best_epoch=None):


	# plot the train and test metrics
	plt.close()
	fig, axes = plt.subplots(2, 3, figsize=(20, 10))

	# plot losses
	mt.plot_metric(axes[0, 0], keys=["train_loss", "valid_loss", "test_loss"], y_log='log')
	mt.plot_metric(axes[0, 1], keys=["auc_roc_train", "auc_roc_valid", "auc_roc_test"])
	mt.plot_metric(axes[0, 2], keys=["f1_class_macro_train", "f1_class_macro_valid", "f1_class_macro_test"])
	mt.plot_metric(axes[1, 0], keys=["lr"], y_log='log')

	if best_epoch is not None:
		axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--')
		axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--')
		axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--')
		axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--')

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


	# load trials
	dataloaders = load_data(args)
	"""
	train      Other                <class 'torch.utils.trials.dataloader.DataLoader'>             N/A N/A       
	valid      Other                <class 'torch.utils.trials.dataloader.DataLoader'>             N/A N/A       
	test       Other                <class 'torch.utils.trials.dataloader.DataLoader'>             N/A N/A  
	"""

	# 4) Model 생성
	model = IAT_model(args)
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
		aucs = calculate_roc_aucs(train_data, valid_data, test_data)

		mt.update(epoch, **losses, **f1s, **aucs, **lr)
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
	# save metrics
	df_metrics = mt.generate_df()
	metric_cfg = {
		"valid_loss": {
			"weight": 0.3,
			"direction": "min",
			"log": True,
			"smooth": {"method": "ema", "kw": {"alpha": 0.2}}
		},
		"auc_roc_valid": {
			"weight": 0.5,
			"direction": "max",
			"smooth": {"method": "ema", "kw": {"alpha": 0.2}}
		},
		"f1_class_macro_valid": {
			"weight": 0.2,
			"direction": "max",
			"smooth": {"method": "ema", "kw": {"alpha": 0.2}}
		},
	}
	score_series, best_idx = compute_weighted_score(df_metrics, metric_cfg)
	df_metrics["score"] = score_series
	args["best_score"] = df_metrics["auc_roc_test"].iloc[best_idx]
	df_metrics.to_csv(f"{tm.trial_dir}/train_test_metrics.csv", index=False)
	save_df_as_npz(df_metrics, f"{tm.trial_dir}/train_test_metrics.npz")


	# save plot
	plot_losses(tm, mt, best_epoch=best_idx)


	# save hyperparameters
	save_json(args, f"{tm.trial_dir}/hyperparameters.json")

	return args["best_score"]