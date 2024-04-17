"""
다음과 같은 file 이 있어
/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원/PSG group 2 (PD without RBD)/edf1/raw_microvolt.h5


내가 이 "raw_microvolt.h5" 을 열어서 data 라는 np.array 를 만들고 이것을 저장하려고 해.
그런데 이것을 ./대전성모병원/PSG group 2 (PD without RBD)/edf1/data.npy 로 저장을 하고 싶어.

나는 이 작업을 하는 function 이나 class 에 다음 3개의 정보를 줘서 해결하고 싶어.
- data 위치: "/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원/PSG group 2 (PD without RBD)/edf1/raw_microvolt.h5"
- 필요없는 path: /media/sac/WD4T/Projects_backup/eeg_data/RBD/
- 저장하고자 하는 위치: './'
- 저장하자 하는 파일 이름: data.npy
- data: np.array

이 function 혹은 class 는 "data 위치" 에서 "필요없는 path" 를 지우고 대신 "저장하고자 하는 위치" 를 넣고, 파일이름을 붙여.
그리고 directory 가 존재하지 않으면 reculsive 하게 만들어서 저장하게 하고 싶어.

그러한 function 혹은 class 를 만들어줘.

이 모듈은 내부에서 h5 파일을 열 필요 없고, 그냥 path 의 processing 과 저장만 담당하면 돼

"""


import os
from pathlib import Path
from typing import Union

class PathProcessor:
	def __init__(self, data_path: str, unnecessary_path: str, save_dir: str, save_filename: str):
		self.data_path = Path(data_path)
		self.unnecessary_path = Path(unnecessary_path)
		self.save_dir = Path(save_dir)
		self.save_filename = save_filename

	def process_and_save(self, data: Union[list, tuple, set, dict]) -> None:
		try:
			self._validate_paths()
			processed_path = self._remove_unnecessary_path()
			save_path = self._create_save_path(processed_path)
			self._save_data(save_path, data)
		except (ValueError, OSError) as e:
			print(f"Error occurred: {str(e)}")

	def _validate_paths(self) -> None:
		if not self.data_path.is_file():
			raise ValueError(f"Invalid data path: {self.data_path}")
		if not self.unnecessary_path.is_dir():
			raise ValueError(f"Invalid unnecessary path: {self.unnecessary_path}")

	def _remove_unnecessary_path(self) -> Path:
		return self.data_path.relative_to(self.unnecessary_path)

	def _create_save_path(self, processed_path: Path) -> Path:
		save_path = self.save_dir / processed_path.parent / self.save_filename
		save_path.parent.mkdir(parents=True, exist_ok=True)
		return save_path

	def _save_data(self, save_path: Path, data: Union[list, tuple, set, dict]) -> None:
		try:
			import numpy as np
			np.save(str(save_path), data)
		except ImportError:
			raise ImportError("NumPy library is required to save the data.")


"""
if __name__ == '__main__':
	data_location = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원/PSG group 2 (PD without RBD)/edf1/raw_microvolt.h5"
	unnecessary_path = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/"
	save_location = "./"
	save_filename = "data.npy"

	# Create an instance of PathProcessor
	path_processor = PathProcessor(data_location, unnecessary_path, save_location, save_filename)

	# Assuming you have already loaded the data from the h5 file
	data = np.array([1, 2, 3, 4, 5])  # Replace with your actual data

	# Process the path and save the data
	path_processor.process_and_save(data)
"""