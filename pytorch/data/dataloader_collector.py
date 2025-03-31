
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util_sac.pytorch.data.print_array_info import print_array_info


class TensorDataset(Dataset):
	def __init__(self, tensor_data):
		self.tensor_data = tensor_data
		self.length = len(next(iter(tensor_data.values())))
		"""
		self.length = 4
		self.tensor_data = 
			Key        Type            Shape                    Memory Dtype     
			----------------------------------------------------------------------
			REM_emg    PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
			REM_mask   PyTorch Tensor  (4, 300, 1, 1, 10)      46.88 KB torch.float32
			...
		"""

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		return {key: value[idx] for key, value in self.tensor_data.items()}


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
		numpy_data = self.to_numpy()
		tensor_data = {}
		for key, value in numpy_data.items():
			tensor_data[key] = torch.tensor(value)
		return tensor_data


	def to_dataloader(self, batch_size, shuffle):
		"""Convert collected data to a PyTorch DataLoader."""
		tensor_data = self.to_tensor()
		dataset = TensorDataset(tensor_data)
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():

	# get only first 3
	df_fn = df_fn.head(50)
	# print_partial_markdown(df_fn)

	# prepare subject list
	dc = dataloader_collector(
		{"REM_emg": "float32",
		 "REM_mask": "float32",
		 "NREM_emg": "float32",
		 "NREM_mask": "float32",
		 "class_rbd": "int64",
		 "class_pd": "int64",
		 "hospital": "int64"}
	)

	for idx, row in df_fn.iterrows():
		print(f"Loading {idx + 1}/{len(df_fn)}: {row['File Path']}", flush=True)

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

	loader = dc.to_dataloader(batch_size=n_batch, shuffle=True)

	# print batch
	"""
	batch = next(iter(loader))
	print_array_info(batch)

	Key        Type            Shape                    Memory Dtype     
	----------------------------------------------------------------------
	REM_emg    PyTorch Tensor  (16, 300, 1, 1, 10)    187.50 KB torch.float32
	REM_mask   PyTorch Tensor  (16, 300, 1, 1, 10)    187.50 KB torch.float32
	NREM_emg   PyTorch Tensor  (16, 700, 1, 1, 10)    437.50 KB torch.float32
	NREM_mask  PyTorch Tensor  (16, 700, 1, 1, 10)    437.50 KB torch.float32
	class_rbd  PyTorch Tensor  (16,)                    0.06 KB torch.int32
	class_pd   PyTorch Tensor  (16,)                    0.06 KB torch.int32
	hospital   PyTorch Tensor  (16, 5)                  0.31 KB torch.int32
	"""

	return loader


if __name__ == "__main__":
	main()
