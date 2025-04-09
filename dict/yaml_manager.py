import argparse
from types import SimpleNamespace

import yaml


def save_yaml(args, filename):
	"""
	'args'가 'argparse.Namespace' 혹은 'types.SimpleNamespace' 형식이면 vars() 함수를 사용하여 dict로 변환하고,
	이미 dict라면 그대로 YAML 파일로 저장한다.
	"""
	if isinstance(args, argparse.Namespace) or isinstance(args, SimpleNamespace):
		data = vars(args)
	elif isinstance(args, dict):
		data = args
	else:
		raise ValueError("지원되지 않는 args 타입이다. 'argparse.Namespace', 'types.SimpleNamespace' 또는 'dict'만 가능하다.")

	with open(filename, "w", encoding="utf-8") as f:
		yaml.safe_dump(data, f, allow_unicode=True)


def load_yaml(filename):
	"""YAML 파일을 읽어서 dict 형태로 불러온다."""
	try:
		with open(filename, "r", encoding="utf-8") as f:
			args_dict = yaml.safe_load(f)
	except FileNotFoundError:
		raise FileNotFoundError(f"파일 {filename} 을(를) 찾을 수 없다.")
	except yaml.YAMLError as e:
		raise ValueError(f"파일 {filename} 의 YAML 파싱에 실패했다: {e}")

	return args_dict
