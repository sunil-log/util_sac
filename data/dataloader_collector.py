
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

	dc = dataloader_collector(
		{"REM_emg": "float32",
		 "REM_mask": "float32",
		 "NREM_emg": "float32",
		 "NREM_mask": "float32",
		 "class_rbd": "int32",
		 "class_pd": "int32",
		 "hospital": "int32"}
	)

	for idx, row in df_fn.iterrows():
		print(f"Loading {idx+1}/{len(df_fn)}: {row['File Path']}", flush=True)

		# load z data
		z = load_reduced_data(row["File Path"], n_REM, n_NREM)
		dc.add_sample(z)
		"""
		print_array_info(z)

		Key        Type            Shape                    Memory Dtype     
		----------------------------------------------------------------------
		REM_emg    NumPy Array     (300, 1, 1, 10)         11.72 KB float32
		REM_mask   NumPy Array     (300, 1, 1, 10)         11.72 KB float32
		NREM_emg   NumPy Array     (700, 1, 1, 10)         27.34 KB float32
		NREM_mask  NumPy Array     (700, 1, 1, 10)         27.34 KB float32
		class_rbd  Other           <class 'numpy.int32'>        N/A N/A       
		class_pd   Other           <class 'numpy.int32'>        N/A N/A       
		hospital   NumPy Array     (5,)                     0.04 KB int64
		"""

	tensor_data = dc.to_tensor()
	"""
	print_array_info(tensor_data)
	REM_emg    PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
	REM_mask   PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
	NREM_emg   PyTorch Tensor  (4, 700, 1, 1, 10)     109.38 KB torch.float32
	NREM_mask  PyTorch Tensor  (4, 700, 1, 1, 10)     109.38 KB torch.float32
	class_rbd  PyTorch Tensor  (4,)                     0.02 KB torch.int32
	class_pd   PyTorch Tensor  (4,)                     0.02 KB torch.int32
	hospital   PyTorch Tensor  (4, 5)                   0.08 KB torch.int32
	"""


if __name__ == "__main__":
	main()
