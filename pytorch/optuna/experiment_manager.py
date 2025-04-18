
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""

import optuna
from typing import List, Dict

from util_sac.pytorch.optuna.get_objective import get_objective
from util_sac.pytorch.optuna.session_manager import train_multiple_sessions


def generate_lr_schedules(
	num_schedules: int,
	total_epochs: int,
	*,
	second_drop_ratio: float = 0.9,
	lr_first: float = 1e-4,
	lr_second: float = 1e-5,
) -> List[Dict[int, float]]:

	"""
	두 번 learning‑rate( lr )가 하락하는 스케줄을 생성한다.
	첫 번째 하락 시점(e1)은 **index/num_schedules × total_epochs**로 선형 동기화시키고,
	두 번째 하락 시점(e2)은 e1 이후 남은 구간의 *second_drop_pct %* 지점이 되도록 결정한다.

	Parameters
	----------
	num_schedules : int
		생성할 스케줄 개수.
	total_epochs : int
		전체 epoch 수.
	second_drop_pct : float, default 10.0
		e1 이후 남은 구간 중 몇 퍼센트 시점에서 두 번째 하락을 적용할지 결정하는 비율.
		예) 10.0 → 남은 기간의 10 %.
	lr_first : float, default 1e‑4
		첫 번째 하락 후 learning‑rate.
	lr_second : float, default 1e‑5
		두 번째 하락 후 learning‑rate.

	Returns
	-------
	List[dict[int, float]]
		각 스케줄을 {epoch: lr} 형태로 보관한 리스트.
		예) [{30: 1e‑4, 45: 1e‑5}, …]

	Notes
	-----
	* e1, e2 는 1‑based 정수 epoch 로 계산한다.
	* e2 ≤ total_epochs 를 보장하며, 계산 결과가 e1 과 겹칠 경우 e1 + 1 로 보정한다.
	"""
	schedules: List[Dict[int, float]] = []

	for idx in range(1, num_schedules + 1):
		# 1) e1: index 에 선형 대응 (가장 앞 schedule 도 최소 epoch 1 보장)
		e1 = max(1, round(idx / num_schedules * total_epochs))

		# 2) e2: e1 이후 remaining 의 second_drop_pct %
		remaining = total_epochs - e1
		e2 = e1 + max(1, round(remaining * second_drop_ratio))

		# 총 epoch를 넘지 않도록 클램프
		e2 = min(e2, total_epochs)

		# e1, e2 가 동일해질 가능성을 한 번 더 방지
		if e2 == e1:
			e2 = min(e1 + 1, total_epochs)

		schedules.append({e1: lr_first, e2: lr_second})

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



