
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.dict.json_manager import save_json, load_json


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


class DataCollector:
	"""
	DataCollector 클래스이다. 이 클래스는 구조화된 데이터를 수집하고,
	수집된 데이터를 numpy array로 변환하거나 .npz 파일 형태로 저장하기 위한 기능을 제공한다.


	디자인 원리:
		- 본 클래스는 record(행) 단위로 데이터를 추가(add_sample)하도록 설계한다.
		  이는 한 번에 key 방향으로 데이터를 합치는 방식에 비해 다소 비효율적으로 보일 수 있다.
		- 그러나 비정형 데이터를 다룰 때는 보통 같은 ID를 가진 데이터가 서로 다른 key를 통해
		  묶이는 경우가 잦다. 이때, key 방향으로 concat을 시도하면 ID가 동일한지 확인하는 로직이
		  필요하게 된다.
		- 따라서 처음부터 ID 방향으로 item(샘플)을 추가함으로써, 중복이나 누락 같은 실수를
		  줄일 수 있으며, 수집된 데이터를 통합 관리하기가 용이해진다.

	구조:
		1) structure (dict): key는 데이터 필드 이름, value는 'float32', 'float64', 'int32', 'int64', 'bool' 중 하나이다.
		2) data (dict): structure에 정의된 key를 기준으로, 각 key에 해당하는 데이터가 리스트로 쌓인다.

	주요 메서드:
		- add_sample(sample): 샘플(행) 단위로 데이터를 추가한다.
		- to_numpy(): 수집된 데이터를 numpy array로 변환한다.
		- save_npz(target_dir): to_numpy() 결과를 data.npz로 저장하고, structure를 data_structure.json으로 저장한다.

	사용 예시:
		structure = {
			'REM_emg': 'float32',
			'stage_label': 'int64',
			'is_sleep': 'bool'
		}
		collector = DataCollector(structure)

		# 데이터 추가
		collector.add_sample({
			'REM_emg': np.array([0.123, 0.234, 0.345], dtype=np.float32),
			'stage_label': 2,
			'is_sleep': True
		})

		# numpy array로 변환
		numpy_data = collector.to_numpy()

		# .npz 파일 저장
		collector.save_npz('some_directory')
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
		객체를 초기화한다.

		:param structure: key가 데이터 필드 이름, value가 허용된 데이터 타입 중 하나('float32', 'float64', 'int32', 'int64', 'bool')인 dict
		"""
		self.structure = {}
		for key, dtype in structure.items():
			if dtype not in self.ALLOWED_TYPES:
				raise ValueError(
					f"Unsupported data type: {dtype}. Allowed types are: {', '.join(self.ALLOWED_TYPES.keys())}")
			self.structure[key] = dtype

		self.data = {key: [] for key in self.structure.keys()}
		"""
		예) 
		self.structure = {'REM_emg': 'float32', ...}
		self.data = {'REM_emg': [], ...}
		"""

	def add_sample(self, sample):
		"""
		샘플(행)을 추가한다.

		:param sample: structure에 정의된 key를 모두 포함하는 dict
		"""
		for key, value in sample.items():
			if key not in self.structure:
				raise KeyError(f"Unexpected key: {key}")
			self.data[key].append(value)

	def to_numpy(self):
		"""
		수집된 데이터를 numpy array로 변환한다.

		:return: {key: numpy_array} 형태의 dict
		"""
		numpy_data = {}
		for key, value_list in self.data.items():
			numpy_type = self.ALLOWED_TYPES[self.structure[key]][0]
			numpy_data[key] = np.array(value_list, dtype=numpy_type)
		return numpy_data

	def save_npz(self, target_dir):
		"""
		수집된 데이터를 to_numpy()로 변환한 뒤, data.npz 형태로 저장하고,
		structure는 data_structure.json 파일로 저장한다.

		:param target_dir: 파일이 저장될 디렉터리 경로
		"""
		os.makedirs(target_dir, exist_ok=True)

		# numpy array로 변환
		numpy_data = self.to_numpy()
		print_array_info(numpy_data)

		# 파일 경로 설정
		npz_path = os.path.join(target_dir, "data.npz")
		json_path = os.path.join(target_dir, "data_structure.json")

		# data.npz 저장
		np.savez(npz_path, **numpy_data)

		# data_structure.json 저장
		save_json(self.structure, json_path)


class TensorDataLoaderMaker:
	"""
	TensorDataLoaderMaker 클래스이다. 이 클래스는 이미 저장된 .npz 파일과
	structure.json 파일(또는 메모리에 이미 로드된 numpy data, structure)을 바탕으로
	to_tensor(), to_dataloader() 기능을 제공한다.

	사용 방법:
		1) data_npz와 structure_json 경로(또는 이미 로드된 data, structure)를 전달하여 객체를 생성한다.
		2) to_tensor()를 호출하여 PyTorch Tensor 형태의 데이터를 얻는다.
		3) to_dataloader(batch_size, shuffle)를 호출하여 PyTorch DataLoader를 얻는다.

	예시:
		maker = TensorDataLoaderMaker(data_npz='some_directory/data.npz',
									  structure_json='some_directory/data_structure.json')

		tensor_data = maker.to_tensor()
		dataloader = maker.to_dataloader(batch_size=16, shuffle=True)
	"""

	def __init__(self, data_npz=None, structure_json=None,
	             loaded_numpy_data=None, loaded_structure=None):
		"""
		:param data_npz: .npz 파일 경로 (예: 'some_directory/data.npz')
		:param structure_json: 구조 정의가 저장된 json 파일 경로 (예: 'some_directory/data_structure.json')
		:param loaded_numpy_data: 이미 메모리에 로드된 numpy data(dict 형태) - data_npz 없이 사용 가능
		:param loaded_structure: 이미 메모리에 로드된 structure(dict 형태) - structure_json 없이 사용 가능
		"""
		if data_npz and structure_json:
			# 파일 로드
			self.data = dict(np.load(data_npz))
			load_json(structure_json)
		elif loaded_numpy_data is not None and loaded_structure is not None:
			# 이미 로드된 numpy data와 structure 사용
			self.data = loaded_numpy_data
			self.structure = loaded_structure
		else:
			raise ValueError("data_npz와 structure_json 경로를 모두 주거나, "
			                 "loaded_numpy_data와 loaded_structure를 모두 제공해야 한다.")

	def to_tensor(self):
		"""
		내부 data(numpy array 형태)를 PyTorch Tensor로 변환한다.

		:return: {key: tensor} 형태의 dict
		"""
		tensor_data = {}
		for key, np_array in self.data.items():
			tensor_data[key] = torch.tensor(np_array)
		return tensor_data

	def to_dataloader(self, batch_size, shuffle):
		"""
		내부 data를 Tensor로 변환한 뒤, PyTorch DataLoader로 감싼다.

		:param batch_size: Batch 크기
		:param shuffle: Shuffle 여부
		:return: torch.utils.data.DataLoader
		"""
		tensor_data = self.to_tensor()
		# tensor_data가 dict이므로, 각 key에 해당하는 value(tensor)만 추출하여
		# (tensor1, tensor2, ...) 형태의 튜플로 구성한다.
		tensors_tuple = tuple(tensor_data.values())
		dataset = TensorDataset(*tensors_tuple)
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
