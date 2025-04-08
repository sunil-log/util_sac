
# -*- coding: utf-8 -*-
"""
Created on  Apr 08 2025

@author: sac
"""


import numpy as np

from util_sac.dict.jsonl_file_manager import jsonl_file_manager
from copy import deepcopy



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



