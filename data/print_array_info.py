

import torch
import numpy as np


def print_array_info(data_dict):
	print(f"{'Key':<10} {'Type':<15} {'Shape':<20} {'Memory':>10} {'Dtype':<10}")
	print("-" * 70)
	for key, data in data_dict.items():
		if isinstance(data, (torch.Tensor, np.ndarray)):
			if isinstance(data, torch.Tensor):
				shape = tuple(data.size())
				memory = data.element_size() * data.nelement()
				dtype = data.dtype
				data_type = "PyTorch Tensor"
			elif isinstance(data, np.ndarray):
				shape = data.shape
				memory = data.nbytes
				dtype = data.dtype
				data_type = "NumPy Array"

			print(f"{key:<10} {data_type:<15} {str(shape):<20} {memory / 1024:>8.2f} KB {dtype}")
		else:
			print(f"{key:<10} {'Other':<15} {str(type(data)):<20} {'N/A':>10} {'N/A':<10}")
