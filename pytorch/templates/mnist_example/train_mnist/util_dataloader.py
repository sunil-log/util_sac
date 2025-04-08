
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

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

from util_sac.pytorch.example_dataset.MNIST_dataloader import prepare_data_combine_train_test
import util_sac.pytorch.dataloader as dlf



def load_data(args):

	# load data
	data = prepare_data_combine_train_test(flatten=True, cnn=True)
	"""
	x          PyTorch Tensor       (70000, 1, 784)            209.35 MB torch.float32
	y          PyTorch Tensor       (70000,)                   546.88 KB torch.int64
	"""

	data = dlf.split_data_into_train_valid_test(
		data=data,
		fold_i=args["fold"]["i"],
		fold_count=args["fold"]["count"],
		fold_seed=args["fold"]["seed"],
		stratify_key="y"
	)
	"""
	print_array_info(data)
	train      Other                <class 'dict'>                   N/A N/A       
	valid      Other                <class 'dict'>                   N/A N/A       
	test       Other                <class 'dict'>                   N/A N/A
	"""

	"""
	print_array_info(data["train"])
	x          PyTorch Tensor       (42000, 1, 784)            125.61 MB torch.float32
	y          PyTorch Tensor       (42000,)                   328.12 KB torch.int64
	"""


	df_count = dlf.label_distribution_table(data, label_col="y")
	print_partial_markdown(df_count)
	"""
	|    | Class   |   count_train |   percent_train |   count_valid |   percent_valid |   count_test |   percent_test |
	|---:|:--------|--------------:|----------------:|--------------:|----------------:|-------------:|---------------:|
	|  0 | 0       |          4142 |         9.8619  |          1380 |         9.85714 |         1381 |        9.86429 |
	|  1 | 1       |          4725 |        11.25    |          1576 |        11.2571  |         1576 |       11.2571  |
	|  2 | 2       |          4194 |         9.98571 |          1398 |         9.98571 |         1398 |        9.98571 |
	|  3 | 3       |          4284 |        10.2     |          1428 |        10.2     |         1429 |       10.2071  |
	|  4 | 4       |          4095 |         9.75    |          1365 |         9.75    |         1364 |        9.74286 |
	|  5 | 5       |          3787 |         9.01667 |          1263 |         9.02143 |         1263 |        9.02143 |
	|  6 | 6       |          4125 |         9.82143 |          1376 |         9.82857 |         1375 |        9.82143 |
	|  7 | 7       |          4377 |        10.4214  |          1458 |        10.4143  |         1458 |       10.4143  |
	|  8 | 8       |          4095 |         9.75    |          1365 |         9.75    |         1365 |        9.75    |
	|  9 | 9       |          4176 |         9.94286 |          1391 |         9.93571 |         1391 |        9.93571 |
	| 10 | Total   |         42000 |       100       |         14000 |       100       |        14000 |      100       |
	"""


	dataloaders = dlf.create_dataloaders(data, batch_size=128, shuffle=True)
	"""
	train      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	valid      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	test       Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A  
	"""

	"""
	batch = next(iter(dataloaders["train"]))
	print_array_info(batch)
	x          PyTorch Tensor       (32, 1, 784)                98.00 KB torch.float32
	y          PyTorch Tensor       (32,)                       256.00 B torch.int64
	"""

	return dataloaders





