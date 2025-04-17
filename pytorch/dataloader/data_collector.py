
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""

import os

import numpy as np
import torch

from util_sac.dict.json_manager import save_json
from util_sac.pytorch.print_array import print_array_info


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
		1) structure (dict): key는 데이터 필드 이름, value는 'float32', 'float64', 'int32', 'int64', 'bool', 'str' 중 하나이다.
		2) str 의 경우 torch에서는 직접적인 dtype이 없으므로 차후 torch 로 변환할 때는 주의해야 한다.
		3) trials (dict): structure에 정의된 key를 기준으로, 각 key에 해당하는 데이터가 리스트로 쌓인다.

	주요 메서드:
		- add_sample(sample): 샘플(행) 단위로 데이터를 추가한다.
		- to_numpy(): 수집된 데이터를 numpy array로 변환한다.
		- save_npz(target_dir): to_numpy() 결과를 trials.npz로 저장하고, structure를 data_structure.json으로 저장한다.

	사용 예시:
		structure = {
			'path': 'str',
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
		'bool': (np.bool_, torch.bool),
		'str': (np.str_, None)  # str 처리 추가 (PyTorch에는 직접적인 str dtype이 없으므로 None)
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
					f"Unsupported trials type: {dtype}. Allowed types are: {', '.join(self.ALLOWED_TYPES.keys())}")
			self.structure[key] = dtype

		self.data = {key: [] for key in self.structure.keys()}
		self.flag_numpy = False
		"""
		예) 
		self.structure = {'REM_emg': 'float32', ...}
		self.trials = {'REM_emg': [], ...}
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
		self.flag_numpy = True
		numpy_data = {}
		for key, value_list in self.data.items():
			numpy_type = self.ALLOWED_TYPES[self.structure[key]][0]
			numpy_data[key] = np.array(value_list, dtype=numpy_type)
		self.data = numpy_data
		return numpy_data

	def save_npz(self, target_dir):
		"""
		수집된 데이터를 to_numpy()로 변환한 뒤, trials.npz 형태로 저장하고,
		structure는 data_structure.json 파일로 저장한다.

		:param target_dir: 파일이 저장될 디렉터리 경로
		"""
		if not self.flag_numpy:
			raise RuntimeError("Data must be converted to numpy array before saving.")

		os.makedirs(target_dir, exist_ok=True)

		# 파일 경로 설정
		npz_path = os.path.join(target_dir, "trials.npz")
		json_path = os.path.join(target_dir, "data_structure.json")

		# trials.npz 저장
		print(f"\n\nSaving trials to {target_dir}")
		print_array_info(self.data)
		np.savez(npz_path, **self.data)

		# data_structure.json 저장
		save_json(self.structure, json_path)



def main():

	pass

if __name__ == "__main__":
	main()
