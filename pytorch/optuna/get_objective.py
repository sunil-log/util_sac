
# -*- coding: utf-8 -*-
"""
Created on  Mar 22 2025

@author: sac
"""

import optuna
from types import SimpleNamespace
import random


"""
from types import SimpleNamespace

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


def train_session(args):
	import random
	return random.random()  # 임의 스코어


def objective(trial: optuna.trial.Trial):
	# 2) sample_params 함수를 통해 dict 획득
	args_dict = sample_params(trial, param_space)

	# 3) SimpleNamespace로 감싸서 사용 (편의를 위해)
	args = SimpleNamespace(**args_dict)

	# 4) 이 args를 활용해 학습/검증
	score = train_session(args)
	return score


def main():
	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=10)

	print("Best value:", study.best_value)
	print("Best params:", study.best_params)


if __name__ == "__main__":
	main()
"""



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


# 예시 사용:
if __name__ == "__main__":
	lr_schedules_50 = generate_lr_schedules(50, 200)
	for schedule in lr_schedules_50[:5]:  # 앞에서 5개만 출력
		print(schedule)


def optuna_sample_params(trial: optuna.trial.Trial, param_space: dict) -> dict:
	"""
	param_space에 정의된 항목을 순회하며, trial.suggest_...()를 자동으로 호출해
	하이퍼파라미터 dict를 생성하여 리턴한다.
	"""
	params = {}
	for name, config in param_space.items():
		param_type = config["type"]

		if param_type == "int":
			# 예) "n_layers": {"type":"int", "low":1, "high":6, "step":1, "log":False}
			low = config["low"]
			high = config["high"]
			step = config.get("step", 1)
			log = config.get("log", False)
			params[name] = trial.suggest_int(name, low, high, step=step, log=log)

		elif param_type == "float":
			# 예) "learning_rate": {"type":"float", "low":1e-5, "high":1e-1, "log":True}
			low = config["low"]
			high = config["high"]
			step = config.get("step", None)  # float에서도 step 지정 가능
			log = config.get("log", False)
			params[name] = trial.suggest_float(name, low, high, step=step, log=log)

		elif param_type == "categorical":
			# 예) "model_type": {"type":"categorical", "choices":["transformer", "rnn", "cnn"]}
			choices = config["choices"]
			params[name] = trial.suggest_categorical(name, choices)

		elif param_type == "bool":
			# optuna에는 직접적인 'suggest_bool'은 없으나, True/False를 categorical로 처리
			# 예) "use_batchnorm": {"type":"bool"}
			params[name] = trial.suggest_categorical(name, [True, False])

		else:
			raise ValueError(f"알 수 없는 타입: {param_type}")

	return params




def get_objective(
		study_info: dict,
		param_space: dict,
		lr_dicts: list,
		train_sessions: callable
):

	"""
	study_info 을 넣기 위한 wrapper 함수
	"""

	def objective(trial: optuna.trial.Trial) -> float:

		"""
		Optuna의 objective 함수
		"""

		# sample hyperparameters
		args_dict = optuna_sample_params(trial, param_space)
		args = SimpleNamespace(**args_dict)
		args.lr_dict = lr_dicts[args.lr_dict_idx]

		# 이제 train_session에 넘겨서 학습
		args.study_name = study_info["study_name"]
		args.optuna_trial_index = trial.number
		args.trial_name = f"{study_info['study_name']}__Trial_{trial.number}"  # 원하는 형식으로
		args.db_dir = study_info["db_dir"]

		# run multiple train sessions
		score = train_sessions(args)
		return score

	return objective

