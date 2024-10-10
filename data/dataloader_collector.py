
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""


import numpy as np
import torch



class dataloader_collector:

	ALLOWED_TYPES = {
		'float32': (np.float32, torch.float32),
		'float64': (np.float64, torch.float64),
		'int32': (np.int32, torch.int32),
		'int64': (np.int64, torch.int64),
		'bool': (np.bool_, torch.bool)
	}

	def __init__(self, structure):
		"""
		Initialize the DataCollector with a specified structure.

		:param structure: A dictionary where keys are data field names and values are their types.
						  Allowed types: 'float32', 'float64', 'int32', 'int64', 'bool'
		"""
		self.structure = {}
		for key, dtype in structure.items():
			if dtype not in self.ALLOWED_TYPES:
				raise ValueError(
					f"Unsupported data type: {dtype}. Allowed types are: {', '.join(self.ALLOWED_TYPES.keys())}")
			self.structure[key] = dtype

		self.data = {key: [] for key in self.structure.keys()}
		"""
		self.structure = {'REM_emg': 'float32', ...}
		self.data = {'REM_emg': [], ...}
		"""


	def add_sample(self, sample):
		"""Add a sample to the collector."""
		for key, value in sample.items():
			if key not in self.structure:
				raise KeyError(f"Unexpected key: {key}")
			self.data[key].append(value)

	def to_numpy(self):
		"""Convert collected data to numpy arrays."""
		numpy_data = {}
		for key, value_list in self.data.items():
			numpy_type = self.ALLOWED_TYPES[self.structure[key]][0]
			numpy_data[key] = np.array(value_list, dtype=numpy_type)
		return numpy_data


	def to_tensor(self):
		"""Convert collected data to PyTorch tensors."""
		tensor_data = {}
		for key, value_list in self.data.items():
			torch_type = self.ALLOWED_TYPES[self.structure[key]][1]
			tensor_data[key] = torch.tensor(value_list, dtype=torch_type)
		return tensor_data


def main():
	pass

if __name__ == "__main__":
	main()
