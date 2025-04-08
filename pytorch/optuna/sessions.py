
# -*- coding: utf-8 -*-
"""
Created on  Apr 08 2025

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


from util_sac.dict.jsonl_file_manager import jsonl_file_manager
from util_sac.pytorch.optuna.get_objective import get_objective
from copy import deepcopy
from util_sac.pytorch.optuna.get_objective import generate_lr_schedules



def check_args_integrity(args):

	if args["fold"]["mode"] == "single":
		assert isinstance(args["fold"]["i"], int), (
			"args['fold']['mode']가 'single'인 경우, args['fold']['i']는 int이어야 한다."
		)
		assert args["fold"]["i"] < args["fold"]["count"], (
			"args['fold']['mode']가 'single'인 경우, args['fold']['i']가 "
			"args['fold']['count']보다 작아야 한다."
		)
	elif args["fold"]["mode"] == "all":
		assert args["fold"]["i"] is None, (
			"args['fold']['mode']가 'all'인 경우, args['fold']['i']는 None 이어야 한다."
		)



def train_multiple_sessions(train_session, args):

	# 1) 인자 무결성 확인
	check_args_integrity(args)
	args_copy = deepcopy(args)

	# 2) 결과 저장용 파일
	fn_results = f"{args_copy['db_dir']}/scores.jsonl"
	jm = jsonl_file_manager(fn_results)

	# 3) 세션 인덱스 설정
	mode = args_copy["fold"]["mode"]
	if mode == "all":
		indices = range(args_copy["fold"]["count"])
	elif mode == "single":
		indices = [args_copy["fold"]["i"]]
	else:
		raise ValueError("args['fold']['mode']가 'single' 또는 'all'이어야 한다.")

	# 4) 세션 반복
	list_score = []
	for i in indices:
		local_args = deepcopy(args_copy)
		local_args["fold"]["i"] = i
		local_args["trial_name"] = f"{local_args['trial_name']}__session_{i}"
		score = train_session(local_args)
		list_score.append(score)

	# 5) 스코어 기록 및 평균 계산
	d = deepcopy(args_copy)
	for i, score in enumerate(list_score):
		d[f"score_{i}"] = score
	d["mean_score"] = np.mean(list_score)
	jm.write_line(d)

	return d["mean_score"]


def run_session(config, train_session):

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






def main():
	pass

if __name__ == "__main__":
	main()
