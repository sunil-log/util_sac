
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""


import json
import argparse
from types import SimpleNamespace

def save_json(args, filename):
	"""
	'args'가 'argparse.Namespace' 혹은 'types.SimpleNamespace' 형식이면 vars() 함수를 사용하여 dict로 변환하고,
	이미 dict라면 그대로 JSON 파일로 저장한다.
	"""
	if isinstance(args, argparse.Namespace) or isinstance(args, SimpleNamespace):
		data = vars(args)
	elif isinstance(args, dict):
		data = args
	else:
		raise ValueError("지원되지 않는 args 타입이다. 'argparse.Namespace', 'types.SimpleNamespace' 또는 'dict'만 가능하다.")

	with open(filename, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filename):
	"""JSON 파일을 읽어서 dict 형태로 불러온다."""
	try:
		with open(filename, "r", encoding="utf-8") as f:
			args_dict = json.load(f)
	except FileNotFoundError:
		raise FileNotFoundError(f"파일 {filename} 을(를) 찾을 수 없다.")
	except json.JSONDecodeError:
		raise ValueError(f"파일 {filename} 의 JSON 파싱에 실패했다.")

	return args_dict
