

from pathlib import Path
from datetime import datetime
from util_sac.sys.zip_df import backup_keywords


import os
import re
from datetime import datetime

def validate_trial_name(trial_name):
	# Ubuntu 파일 시스템에서 허용되지 않는 문자 체크
	invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
	if re.search(invalid_chars, trial_name):
		raise ValueError(f"Trial name '{trial_name}' contains invalid characters for Ubuntu filesystem.")

	# 파일 이름 길이 체크 (Ubuntu의 최대 파일 이름 길이는 255바이트)
	if len(trial_name.encode('utf-8')) > 255:
		raise ValueError(f"Trial name '{trial_name}' is too long for Ubuntu filesystem.")

	# 예약어 체크
	reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
	if trial_name.upper() in reserved_names:
		raise ValueError(f"Trial name '{trial_name}' is a reserved name and cannot be used.")

	# 파일 이름이 '.' 또는 '..'로 시작하는지 체크
	if trial_name.startswith('.') or trial_name.startswith('..'):
		raise ValueError(f"Trial name '{trial_name}' cannot start with '.' or '..'.")



class trial_manager:
	"""
	trial_manager 클래스는 실험 디렉토리와 하위 디렉토리를 생성하고 관리하는 기능을 제공합니다.

	Args:
		sub_dir_list (list): 생성할 하위 디렉토리 이름 목록
		trial_name (str, optional): 실험 이름. 기본값은 None이며, 이 경우 사용자 입력으로 설정됩니다.

	Attributes:
		trial_dir (Path): 생성된 실험 디렉토리 경로
		sub_dir_dict (dict): 생성된 하위 디렉토리 경로 딕셔너리

	Methods:
		__getitem__(key): 인덱스 또는 키를 사용하여 하위 디렉토리 경로에 접근합니다.

	사용 예시:
		sub_dir_list = ["weights", "reconstruction", "latent_space"]
		tm = trial_manager(sub_dir_list, trial_name="example")

		plt.savefig(f"{tm.trial_dir}/{tm.date_prefix}__train_test_metrics.png")
		plt.savefig(f"{tm['reconstruction']}/{tm.date_prefix}__epoch_{epoch}.png")
		torch.save(model.state_dict(), tm['weights'] / f"{tm.date_prefix}__model_{epoch}.pt")
	"""

	def __init__(self, sub_dir_list, trial_name="test"):

		# validate trial name
		validate_trial_name(trial_name)

		# create trial directory
		self.date_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		self.trial_dir = self.__create_trial_dir(trial_name)
		self.sub_dir_dict = self.__create_sub_dirs(sub_dir_list)
		self.source_path = self.trial_dir / f"{self.date_prefix}__{trial_name}.zip"
		"""
		self.trial_dir: trials/2024-10-09_17-02-16__test
		self.sub_dir_dict: 
			{'weights': 'trials/2024-10-09_17-02-16__test/weights',
			 ...}
		"""


		# backup source code
		key_in = [".py", ".m", ".sh", ".txt", "Dockerfile"]
		key_out = [".pyc", ".png", ".npy", "__pycache__", '.npz', '.pkl', '.zip', '.mat']
		backup_keywords(self.source_path, key_in, key_out, src_loc=".")

	def get_trial_dir(self):
		return self.trial_dir

	def __create_trial_dir(self, trial_name):
		base_dir = Path("./trials")
		base_dir.mkdir(exist_ok=True)

		dir_name = f"{self.date_prefix}__{trial_name}"
		trial_dir = base_dir / dir_name
		trial_dir.mkdir()

		return trial_dir

	def __create_sub_dirs(self, sub_dir_list):
		sub_dir_dict = {}
		for sub_dir in sub_dir_list:
			sub_dir_path = self.trial_dir / sub_dir
			sub_dir_path.mkdir()
			sub_dir_dict[sub_dir] = sub_dir_path

		return sub_dir_dict

	def __getitem__(self, key):
		if isinstance(key, int):
			if 0 <= key < len(self.sub_dir_dict):
				return list(self.sub_dir_dict.values())[key]
			else:
				raise IndexError(f"Index {key} is out of range.")
		elif isinstance(key, str):
			if key in self.sub_dir_dict:
				return self.sub_dir_dict[key]
			else:
				raise KeyError(f"Key '{key}' not found in sub_dir_dict.")
		else:
			raise TypeError(f"Key must be either int or str, not {type(key)}.")