
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""

import optuna

from util_sac.pytorch.optuna.get_objective import get_objective, generate_lr_schedules
from util_sac.pytorch.optuna.session_manager import train_multiple_sessions



def lr_dict_from_args_static(args):
	# lr dicts
	if "lr_dict" in args:
		lr_dict = args["lr_dict"]
		lr_dict = {key: float(value) for key, value in lr_dict.items()}
	else:
		lr_dict = {5: 1e-4, 100: 1e-5}

	return lr_dict

def execute_experiment(config, train_session):



	if 'optimize' not in config or config['optimize'] is None:

		"""
		Only Static section
		"""
		args = config["static"]
		args["lr_dict"] = lr_dict_from_args_static(args)

		# train_session
		score = train_multiple_sessions(train_session, args)

		# print score
		print("Single Trial Score:", score)


	else:
		"""
		Optuna optimize section
		"""

		# get params
		args_static = config["static"]
		param_space = config["optimize"]


		# lr dict related
		if "lr_dict_idx" in param_space:
			"""
			lr_dict_idx 가 optimize 에 포함된 경우
			
			  lr_dict_idx:
			    type: "int"
			    low: 0
			    high: 50
			    log: false
			"""
			lr_dicts = generate_lr_schedules(
				num_schedules=param_space["lr_dict_idx"]["high"]+1,
				total_epochs=args_static["n_epoch"]
			)

		else:
			# lr 을 optimize 하지 않을 경우
			args_static["lr_dict_idx"] = 0
			lr_dicts = [lr_dict_from_args_static(args_static)]


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



