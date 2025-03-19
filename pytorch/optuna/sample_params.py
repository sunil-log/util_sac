import optuna


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



def sample_params(trial: optuna.trial.Trial, param_space: dict) -> dict:
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


