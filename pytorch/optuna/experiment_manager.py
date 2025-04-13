
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""

import optuna

from util_sac.pytorch.optuna.get_objective import get_objective, generate_lr_schedules
from util_sac.pytorch.optuna.session_manager import train_multiple_sessions


def execute_experiment(config, train_session):


	if 'optimize' not in config or config['optimize'] is None:

		"""
		Only Static section
		"""
		args = config["static"]

		# lr dicts
		if "lr" in args and isinstance(args["lr"], dict) and "lr_dict" in args["lr"]:
			args["lr_dict"] = args["lr"]["lr_dict"]
			args["lr_dict"] = {key: float(value) for key, value in args["lr_dict"].items()}
		else:
			args["lr_dict"] = {50: 1e-4, 100: 1e-5}

		# train_session
		score = train_multiple_sessions(train_session, args)

		# print score
		print("Single Trial Score:", score)


	else:
		"""
		lr_dicts 생성
		"""
		lr_dicts = generate_lr_schedules(
			num_schedules=50,
			total_epochs=config["static"]["n_epoch"]
		)


		"""
		Optuna optimize section
		"""

		# get params
		args_static = config["static"]
		param_space = config["optimize"]

		# add params
		args_static["db_path"] = f"{args_static['db_dir']}/study.db"
		study_info = {
			"study_name": args_static["study_id"],
			"db_dir": args_static["db_dir"],
			"db_path": args_static["db_path"],
		}

		# 2) study_name을 이용해 study 생성
		study = optuna.create_study(
			study_name=study_info["study_name"],
			storage=f"sqlite:///{study_info['db_path']}",
			load_if_exists=True,
			direction="maximize"
		)

		# get_objective(study_name)로부터 objective 함수를 얻어서 optimize
		objective_func = get_objective(
			args_static=args_static,
			param_space=param_space,
			lr_dicts=lr_dicts,
			train_session=train_session
		)
		study.optimize(objective_func, n_trials=args_static["n_trial"])

		print("Best value:", study.best_value)
		print("Best params:", study.best_params)



