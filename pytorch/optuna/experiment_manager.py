
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""

import optuna

from util_sac.pytorch.optuna.get_objective import get_objective
from util_sac.pytorch.optuna.session_manager import train_multiple_sessions

def generate_lr_schedules(num_schedules: int, total_epochs: int):
	"""
	두 번 학습률이 떨어지는 스케줄을 num_schedules만큼 랜덤으로 생성한다.
	각 스케줄은 {e1: 1e-4, e2: 1e-5} 형태의 딕셔너리로, e1 < e2 < total_epochs를 만족한다.

	Args:
		num_schedules (int): 생성할 스케줄 개수
		total_epochs (int): 전체 에폭 수

	Returns:
		List[dict]: 두 번에 걸쳐 학습률이 하락하는 스케줄 딕셔너리의 리스트
	"""
	schedules = []
	for _ in range(num_schedules):
		# e1과 e2는 1 ~ total_epochs-1 범위에서 선택하고, e1 < e2 가 되도록 설정
		e1 = random.randint(1, total_epochs - 1)
		e2 = random.randint(e1 + 1, total_epochs)  # e2는 e1보다 무조건 커야 함

		schedule = {e1: 1e-4, e2: 1e-5}
		schedules.append(schedule)
	return schedules



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



