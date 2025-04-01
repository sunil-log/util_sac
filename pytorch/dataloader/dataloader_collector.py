
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util_sac.pytorch.data.print_array import print_array_info


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
	"""
	dataloader_collector 클래스이다. 이 클래스는 구조화된 데이터를 수집하고,
	수집된 데이터를 numpy array, PyTorch Tensor, 혹은 PyTorch DataLoader로
	변환하기 위한 기능을 제공한다. 각 데이터 필드는 사전에 정의된 structure에 따라
	특정 Data Type으로만 수집될 수 있다.

	디자인 원리:
	    - 본 클래스는 record(행) 단위로 데이터를 추가(add_sample)하도록 설계한다.
	      이는 한 번에 key 방향으로 데이터를 합치는 방식에 비해 다소 비효율적으로 보일 수 있다.
	    - 그러나 비정형 데이터를 다룰 때는 보통 같은 ID를 가진 데이터가 서로 다른 key를 통해
	      묶이는 경우가 잦다. 이때, key 방향으로 concat을 시도하면 ID가 동일한지 확인하는 로직이
	      필요하게 된다.
	    - 따라서 처음부터 ID 방향으로 item(샘플)을 추가함으로써, 중복이나 누락 같은 실수를
	      줄일 수 있으며, 수집된 데이터를 통합 관리하기가 용이해진다.

	사용 방법:
	    1) 객체 생성 시, structure라는 dictionary를 입력받는다. 예) {'REM_emg': 'float32', 'stage_label': 'int64'}
	       - ALLOWED_TYPES에 정의된 타입만 사용할 수 있다.
	    2) add_sample(sample) 메서드를 통해 데이터를 추가한다.
	       - sample은 structure에서 정의된 key와 일치하는 field들을 포함해야 하며,
	         각 field에 해당하는 값들이 리스트 형태로 내부에 순차적으로 저장된다.
	    3) to_numpy() 메서드를 호출하면, 수집된 데이터를 numpy array로 변환한 dictionary를 반환한다.
	    4) to_tensor() 메서드를 호출하면, 수집된 데이터를 PyTorch Tensor로 변환한 dictionary를 반환한다.
	    5) to_dataloader(batch_size, shuffle) 메서드를 호출하면, 내부 데이터를 PyTorch DataLoader로 변환한다.
	       - batch_size와 shuffle 파라미터로 미니배치 단위 및 데이터 셔플 여부를 설정할 수 있다.

	예시:
	    structure = {
	        'REM_emg': 'float32',
	        'stage_label': 'int64',
	        'is_sleep': 'bool'
	    }
	    collector = dataloader_collector(structure)

	    # float 값뿐 아니라, numpy array 형태의 float 데이터도 추가 가능하다.
	    collector.add_sample({
	        'REM_emg': np.array([0.123, 0.234, 0.345], dtype=np.float32),
	        'stage_label': 2,
	        'is_sleep': True
	    })
	    collector.add_sample({
	        'REM_emg': np.array([0.456, 0.567, 0.678], dtype=np.float32),
	        'stage_label': 3,
	        'is_sleep': False
	    })

	    # numpy array로 변환
	    numpy_data = collector.to_numpy()

	    # PyTorch Tensor로 변환
	    tensor_data = collector.to_tensor()

	    # PyTorch DataLoader로 변환
	    dataloader = collector.to_dataloader(batch_size=2, shuffle=True)

	주의 사항:
	    - add_sample()로 전달되는 dictionary의 key는 structure에서 정의된 key와 완전히 일치해야 한다.
	    - 잘못된 Data Type의 값을 전달할 경우, 내부적으로 numpy나 PyTorch 변환 시 에러가 발생할 수 있다.
	    - to_dataloader() 내부에서는 TensorDataset(tensor_data) 방식을 사용한다.
	      추후 멀티모달 데이터를 처리해야 하는 경우, Dataset 클래스를 확장하거나
	      별도의 Dataset 클래스를 정의하여 필요한 형태로 변형하는 것이 좋다.
	"""
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
