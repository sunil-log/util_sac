
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.sys.date_format import add_timestamp_to_string
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette



def execute_experiment(config, train_session):

	# lr_dicts 생성
	lr_dicts = generate_lr_schedules(
		num_schedules=50,
		total_epochs=config["static"]["n_epoch"]
	)


	if 'optimize' not in config or config['optimize'] is None:

		"""
		Only Static section
		"""

		args = config["static"]
		args["lr_dict"] = lr_dicts[args["lr_dict_idx"]]
		score = train_multiple_sessions(train_session, args)

		print("Single Trial Score:", score)


	else:
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
		study.optimize(objective_func, n_trials=args_static["n_trials"])

		print("Best value:", study.best_value)
		print("Best params:", study.best_params)



