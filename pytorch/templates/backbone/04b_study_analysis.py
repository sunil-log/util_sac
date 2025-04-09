
# -*- coding: utf-8 -*-
"""
Created on  Mar 27 2025

@author: sac
"""


import numpy as np
import optuna
import pandas as pd

from util_sac.dict.json_manager import load_json
from util_sac.pandas.print_df import print_partial_markdown


def main():
	d = load_json('/sac/mysql.json')
	"""
	{
		'ip': '34.46.56.108', 
		'port': '3306', 
		'db_name': 'optuna_db', 
		'id': 'root', 
		'pass': 'xxx'
	}
	"""

	# 이미 MySQL 등에 저장된 study를 load한다:
	study = optuna.load_study(
		study_name="16-11-30__study_name",
		storage=f"mysql+pymysql://{d['id']}:{d['pass']}@{d['ip']}:{d['port']}/{d['db_name']}"
	)

	# Trial 정보들을 DataFrame으로 불러온다
	df = study.trials_dataframe()
	print_partial_markdown(df)
	"""
	|    |   number |      value | datetime_start      | datetime_complete   | duration        |   params_input_dim |   params_lr_dict_idx |   params_n_head |   params_q_dim | state    |
	|---:|---------:|-----------:|:--------------------|:--------------------|:----------------|-------------------:|---------------------:|----------------:|---------------:|:---------|
	|  0 |        0 | 0.133748   | 2025-03-26 16:07:37 | 2025-03-26 16:07:48 | 0 days 00:00:11 |                  4 |                   27 |              32 |              4 | COMPLETE |
	|  1 |        1 | 0.370564   | 2025-03-26 16:07:57 | 2025-03-26 16:08:09 | 0 days 00:00:12 |                128 |                    0 |             128 |              8 | COMPLETE |
	|  2 |        2 | 0.287456   | 2025-03-26 16:08:18 | 2025-03-26 16:08:29 | 0 days 00:00:11 |                128 |                   25 |               2 |              8 | COMPLETE |
	|  3 |        3 | 0.854703   | 2025-03-26 16:08:38 | 2025-03-26 16:08:49 | 0 days 00:00:11 |                  2 |                   18 |               2 |              2 | COMPLETE |
	|  4 |        4 | 0.790882   | 2025-03-26 16:08:58 | 2025-03-26 16:09:10 | 0 days 00:00:12 |                128 |                   11 |              16 |              4 | COMPLETE |
	"""


if __name__ == "__main__":
	main()
