"""
기능:
- 원본 데이터 파일의 경로에서 필요없는 부분을 제거하고, 원하는 위치에 데이터 파일을 저장합니다.
- 지정된 디렉토리 구조가 존재하지 않는 경우, 필요에 따라 디렉토리를 생성합니다.
- numpy array 형식의 데이터를 .npy 파일로 저장합니다.

사용 방법:
1. 데이터 파일의 전체 경로, 제거하고 싶은 경로 부분, 저장하고자 하는 위치 및 파일 이름을 지정합니다.
2. 데이터(numpy array)와 함께 이 정보를 함수나 클래스에 전달하여 데이터를 적절한 위치에 저장합니다.

예시:
    # data.npy 파일을 './대전성모병원/PSG group 2 (PD without RBD)/edf1/'에 저장
    data_path = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원/PSG group 2 (PD without RBD)/edf1/raw_microvolt.h5"
    unwanted_path = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/"
    save_location = './'
    file_name = 'data.npy'
    save_data_as_npy(data, data_path, unwanted_path, save_location, file_name)

이 모듈은 파일 읽기 기능을 포함하지 않으며, 주어진 numpy array 데이터를 받아 path 를 처리하고 저장하는 역할만을 합니다.
"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Callable
import numpy as np
from datetime import datetime
import glob

class DataProcessor(ABC):
	@abstractmethod
	def process_data(self, file_path: Path) -> Union[list, tuple, set, dict, np.array]:
		pass


class ConcreteDataProcessor(DataProcessor):
	def process_data(self, file_path: Path) -> Union[list, tuple, set, dict, np.array]:
		# Implement the data processing logic here
		# This is just a placeholder example
		data = np.array([1, 2, 3, 4, 5])
		return data

class PathProcessor:
	def __init__(self, data_path: str, unnecessary_path: str, save_dir: str, save_filename: str, data_description: str, processor: DataProcessor):
		self.data_path = Path(data_path)
		self.unnecessary_path = Path(unnecessary_path)
		self.save_dir = Path(save_dir)
		self.save_filename = save_filename
		self.data_description = data_description
		self.processor = processor

	def process_and_save(self) -> None:
		try:
			self._validate_paths()
			processed_path = self._remove_unnecessary_path(self.data_path)
			save_path = self._create_save_path(processed_path)
			if not save_path.exists():
				data = self.processor.process_data(self.data_path)
				self._save_data(save_path, data)
				self._save_data_description(save_path)
			else:
				print(f"Skipping processing for {self.data_path} as {save_path} already exists.")


		except (ValueError, OSError) as e:
			print(f"Error occurred: {str(e)}")

	def _validate_paths(self) -> None:
		if not self.data_path.is_file():
			raise ValueError(f"Invalid data path: {self.data_path}")
		if not self.unnecessary_path.is_dir():
			raise ValueError(f"Invalid unnecessary path: {self.unnecessary_path}")


	def _get_file_paths(self) -> list[Path]:
		return [Path(file_path) for file_path in glob.glob(str(self.data_dir / "**" / "*.h5"), recursive=True)]

	def _remove_unnecessary_path(self, file_path: Path) -> Path:
		return file_path.relative_to(self.unnecessary_path)

	def _create_save_path(self, processed_path: Path) -> Path:
		save_path = self.save_dir / processed_path.parent / self.save_filename
		save_path.parent.mkdir(parents=True, exist_ok=True)
		return save_path

	def _save_data(self, save_path: Path, data: Union[list, tuple, set, dict, np.array]) -> None:
		try:
			np.save(str(save_path), data)
		except ImportError:
			raise ImportError("NumPy library is required to save the data.")
		print(f"\033[92mData saved to: {save_path}\033[0m")

	def _save_data_description(self, save_path: Path) -> None:
		description_path = save_path.with_name(f"{self.save_filename}_description.txt")
		if description_path.exists():
			return
		current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		description_content = f"Time: {current_time}\nFilename: {self.save_filename}\nDescription: {self.data_description}"
		with open(description_path, 'w') as file:
			file.write(description_content)

# Usage example
if __name__ == '__main__':
	data_location = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원/PSG group 2 (PD without RBD)/edf1/"
	unnecessary_path = "/media/sac/WD4T/Projects_backup/eeg_data/RBD/"
	save_location = "./"
	save_filename = "data.npy"
	data_description = "This is a sample data description."

	# Create an instance of ConcreteDataProcessor
	data_processor = ConcreteDataProcessor()

	# Create an instance of PathProcessor
	path_processor = PathProcessor(data_location, unnecessary_path, save_location, save_filename, data_description, data_processor)

	# Process the files and save the data
	path_processor.process_and_save()