
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""


import json

def save_args(args, filename):
	"""args를 dict로 변환하여 JSON 파일로 저장한다."""
	with open(filename, "w") as f:
		json.dump(vars(args), f, indent=4)

def load_args(filename):
	"""JSON 파일을 읽어서 dict 형태로 불러온다."""
	with open(filename, "r") as f:
		args_dict = json.load(f)
	return args_dict
