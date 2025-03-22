
# -*- coding: utf-8 -*-
"""
Created on  Mar 22 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.sys.date_format import add_timestamp_to_string
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.data.print_array_info import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette


from util_sac.pytorch.trainer.load_metrics import load_hyperparams



def main():
	bast_path = './trials/ID_124024__study_name'
	df = load_hyperparams(bast_path)
	print_partial_markdown(df)
	"""
	|     |   input_dim |   n_head |   q_dim |   lr_dict_idx | lr_dict                      | study_name            |   optuna_trial_index | trial_name                                 | db_dir                         |
	|----:|------------:|---------:|--------:|--------------:|:-----------------------------|:----------------------|---------------------:|:-------------------------------------------|:-------------------------------|
	|   0 |         128 |       32 |      16 |             1 | {'50': 0.0001, '80': 1e-05}  | ID_124024__study_name |                   29 | ID_124024__study_name__Trial_29__session_2 | ./trials/ID_124024__study_name |
	|   1 |          16 |        4 |     128 |             2 | {'99': 0.0001, '100': 1e-05} | ID_124024__study_name |                   80 | ID_124024__study_name__Trial_80__session_1 | ./trials/ID_124024__study_name |
	|   2 |         128 |        2 |      64 |             1 | {'50': 0.0001, '80': 1e-05}  | ID_124024__study_name |                   64 | ID_124024__study_name__Trial_64__session_0 | ./trials/ID_124024__study_name |
	|   3 |         128 |       64 |      32 |             1 | {'50': 0.0001, '80': 1e-05}  | ID_124024__study_name |                   72 | ID_124024__study_name__Trial_72__session_1 | ./trials/ID_124024__study_name |
	"""
	exit()

if __name__ == "__main__":
	main()
