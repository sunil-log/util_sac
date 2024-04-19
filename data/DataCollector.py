from typing import List, Dict, Union
import torch
import numpy as np


"""
DataCollector 는 다음과 같은 code 를 짧게 줄일 수 있다.

Without DataCollector:
	z_all = []
	y_all = []
	
	for subject in self.subjects:
	...
		z_all.append(z_padded)
		y_all.append(y)
	
	z_all = torch.stack(z_all)
	y_all = torch.stack(y_all)
	
	
With DataCollector:
	data_collector = DataCollector()
	
	for subject in self.subjects:
	...
		d = {"z": z_padded, "y": y}
		data_collector.append(d)
	data_collector.stack()
	z_all = data_collector["z"]
"""


class DataCollector:

	"""
	DataCollector는 PyTorch Tensor 또는 NumPy 배열 데이터를 수집하고 스택하는 유틸리티 클래스입니다.

	Attributes:
		data (List[Dict[str, Union[torch.Tensor, np.ndarray]]]): 수집된 데이터를 저장하는 리스트.
		stacked_data (Dict[str, Union[torch.Tensor, np.ndarray]]): 스택된 데이터를 저장하는 딕셔너리.
		is_stacked (bool): 데이터가 스택되었는지 여부를 나타내는 플래그.

	Methods:
		__init__(): DataCollector 인스턴스를 초기화합니다.
		append(item: Dict[str, Union[torch.Tensor, np.ndarray]]) -> None:
			새로운 데이터 항목을 데이터 리스트에 추가합니다.
			데이터가 이미 스택된 경우 RuntimeError가 발생합니다.
		stack() -> None:
			수집된 데이터를 키별로 스택합니다.
			데이터가 이미 스택된 경우 아무 작업도 수행하지 않습니다.
			데이터에 지원되지 않는 데이터 유형이 포함된 경우 ValueError가 발생합니다.
			데이터에 누락된 키가 있는 경우 KeyError가 발생합니다.
		__getitem__(key: str) -> Union[torch.Tensor, np.ndarray]:
			스택된 데이터에서 지정된 키에 해당하는 값을 반환합니다.
			키가 스택된 데이터에 없는 경우 KeyError가 발생합니다.
	"""
	def __init__(self):
		self.data: List[Dict[str, Union[torch.Tensor, np.ndarray]]] = []
		self.stacked_data: Dict[str, Union[torch.Tensor, np.ndarray]] = {}
		self.is_stacked: bool = False

	def append(self, item: Dict[str, Union[torch.Tensor, np.ndarray]]) -> None:
		if self.is_stacked:
			raise RuntimeError("Cannot append data after stacking")
		self.data.append(item)

	def stack(self) -> None:
		if not self.is_stacked:
			for key in self.data[0].keys():
				try:
					if isinstance(self.data[0][key], torch.Tensor):
						self.stacked_data[key] = torch.stack([item[key] for item in self.data])
					elif isinstance(self.data[0][key], np.ndarray):
						self.stacked_data[key] = np.stack([item[key] for item in self.data])
					else:
						raise ValueError(f"Unsupported data type for key '{key}'")
				except KeyError:
					raise KeyError(f"Key '{key}' not found in collected data")
			self.is_stacked = True
			self.data = []  # Clear the original data to free memory

	def __getitem__(self, key: str) -> Union[torch.Tensor, np.ndarray]:
		if key not in self.stacked_data:
			raise KeyError(f"Key '{key}' not found in stacked data. Did you forget to call stack()?")
		return self.stacked_data[key]



# Example usage
if __name__ == '__main__':
	data_collector = DataCollector()

	for subject in [1, 2, 3, 4, 5]:
		loss_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * subject
		loss_2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1]) * subject

		data_collector.append({"loss_1": loss_1, "loss_2": loss_2})

	data_collector.stack()

	print("Stacked loss_1:")
	print(data_collector["loss_1"])

	print("Stacked loss_2:")
	print(data_collector["loss_2"])