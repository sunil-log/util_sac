
# -*- coding: utf-8 -*-
"""
Created on  Mar 22 2025

@author: sac
"""

import random
from copy import deepcopy

import optuna

from util_sac.dict.merge_dicts import deep_update
from util_sac.pytorch.optuna.session_manager import train_multiple_sessions

"""

# 1) 예시용 Search Space (param_space) 설정
param_space = {
	"learning_rate": {
		"type": "float",
		"low": 1e-5,
		"high": 1e-1,
		"log": True
	},
	"n_layers": {
		"type": "int",
		"low": 1,
		"high": 6,
		"step": 1,
		"log": False
	},
	"dropout": {
		"type": "float",
		"low": 0.0,
		"high": 0.5,
		"log": False
	},
	"model_type": {
		"type": "categorical",
		"choices": ["transformer", "rnn", "cnn"]
	},
	"use_batchnorm": {
		"type": "bool"
	}
	# 필요하다면 더 추가 가능
}


"""



def optuna_sample_params(trial: optuna.trial.Trial,
						 param_space: dict,
						 prefix: str = None) -> dict:
	"""
	param_space가 중첩된 구조를 가질 때 이를 재귀적으로 순회하며,
	trial.suggest_*()를 자동으로 호출해 하이퍼파라미터 dict를 생성한다.
	"""

	params = {}

	for name, config in param_space.items():

		# 만약 config 자체가 또 다른 dict인데, 'type'이 정의되어 있으면
		# 그 지점이 최종(leaf) 파라미터라고 간주한다.
		if isinstance(config, dict) and "type" in config:
			# 최종 파라미터 이름을 만들 때 prefix가 있으면 'prefix.name' 형식으로 사용한다.
			param_name = f"{prefix}.{name}" if prefix else name
			param_type = config["type"]

			if param_type == "int":
				low = config["low"]
				high = config["high"]
				step = config.get("step", 1)
				log = config.get("log", False)
				params[name] = trial.suggest_int(param_name, low, high, step=step, log=log)

			elif param_type == "float":
				low = config["low"]
				high = config["high"]
				step = config.get("step", None)
				log = config.get("log", False)
				params[name] = trial.suggest_float(param_name, low, high, step=step, log=log)

			elif param_type == "categorical":
				choices = config["choices"]
				params[name] = trial.suggest_categorical(param_name, choices)

			elif param_type == "bool":
				params[name] = trial.suggest_categorical(param_name, [True, False])

			else:
				raise ValueError(f"알 수 없는 타입: {param_type}")

		# 그렇지 않고, 아직 내부에 더 들어가야 하는(중첩된) 구조이면 재귀적으로 처리한다.
		elif isinstance(config, dict):
			sub_prefix = f"{prefix}.{name}" if prefix else name
			params[name] = optuna_sample_params(trial, config, prefix=sub_prefix)

		else:
			raise ValueError(f"'{name}'에서 잘못된 설정이 감지되었음: {config}")

	return params



def get_objective(
		args_static: dict,
		param_space: dict,
		lr_dicts: list,
		train_session: callable
):

	"""
	study_info 을 넣기 위한 wrapper 함수
	"""

	def objective(trial: optuna.trial.Trial) -> float:

		"""
		Optuna의 objective 함수
		"""

		"""
		{
			'study_name': 'MNIST', 
			'trial_name': 'Building', 
			'n_epoch': 15, 
			'lr_dict_idx': 0, 
			'fold': 
				{'mode': 'all', 'i': None, 'count': 5, 'seed': 42}, 
			'model': 
				{'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10}, 
			'save': 
				{'model': 10, 'plot': 10}, 
			'study_id': '15-18-51__MNIST', 
			'db_dir': './trials/15-18-51__MNIST', 
			'db_path': './trials/15-18-51__MNIST/study.db'
		}
		"""


		# 1) sample_params 함수를 통해 dict 획득
		args_optimize = optuna_sample_params(trial, param_space)
		"""
		{'model': {'hidden_dim': 512}}
		"""

		# 2) merge args_static과 args_optimize
		args = deepcopy(args_static)
		args = deep_update(args, args_optimize)

		# 3) lr_dict_idx를 통해 lr_dicts에서 선택
		args["lr_dict"] = lr_dicts[args["lr_dict_idx"]]

		# Optuna Trial DB에 user attribute로 저장
		trial.set_user_attr("lr_dict", args["lr_dict"])

		# 이제 train_session에 넘겨서 학습
		args["trial_number"] = trial.number
		args["trial_name"] = f"{args['trial_name']}__Trial_{trial.number}"

		# run multiple train sessions
		score = train_multiple_sessions(train_session, args)
		return score

	return objective


