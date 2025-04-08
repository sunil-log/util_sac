
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

	# check args integrity
	check_args_integrity(args)
	args_copy = deepcopy(args)


	# fn results
	fn_results = f"{args['db_dir']}/scores.jsonl"
	jm = jsonl_file_manager(fn_results)

	"""
	run sessions
	"""
	list_score = []
	if args["fold"]["mode"] == "all":
		"""
		train multiple sessions
		"""
		for i in range(args["fold"]["count"]):
			args = deepcopy(args_copy)
			args["fold"]["i"] = i
			args["trial_name"] = f"{args['trial_name']}__session_{i}"
			score = train_session(args)
			list_score.append(score)


	elif args["fold"]["mode"] == "single":
		"""
		train single session
		"""
		args["trial_name"] = f"{args['trial_name']}__session_{args['fold']['i']}"
		score = train_session(args)
		list_score.append(score)


	else:
		raise ValueError(
			"args['fold']['mode']가 'single' 또는 'all'이어야 한다."
		)


	# mean_score
	d = deepcopy(args_copy)
	for i, score in enumerate(list_score):
		d[f"score_{i}"] = score
	d["mean_score"] = np.mean(list_score)
	jm.write_line(d)

	return d["mean_score"]




def main():
	pass

if __name__ == "__main__":
	main()
