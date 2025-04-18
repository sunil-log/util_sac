
# -*- coding: utf-8 -*-
"""
Created on  Mar 01 2025

@author: sac
"""

import time

import numpy as np

from train_iat.util_train_session import train_session
from util_sac.dict.yaml_manager import load_yaml
from util_sac.pytorch.optuna.experiment_manager import execute_experiment
from util_sac.sys.files.dir_manager import create_dir

def main():

	"""
	main
	"""

	# 0) create directory
	trial_dir = "./trials"
	create_dir(trial_dir)

	# 1) load config
	config = load_yaml('config.yaml')
	config["static"]["study_id"] = f"{time.strftime('%H-%M-%S')}__{config['static']['study_name']}"
	config["static"]["db_dir"] = f"{trial_dir}/{config['static']['study_id']}"
	create_dir(config["static"]["db_dir"])

	# 3) run session
	execute_experiment(config, train_session)





if __name__ == "__main__":
	main()