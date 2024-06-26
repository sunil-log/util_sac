

from pathlib import Path
from datetime import datetime
from util_sac.sys.zip_df import backup_keywords


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

		plt.savefig(f"{tm.trial_dir}/train_test_metrics.png")
		plt.savefig(f"{tm['reconstruction']}/epoch_{epoch}.png")
		torch.save(model.state_dict(), tm['weights'] / f"model_{epoch}.pt")
	"""

	def __init__(self, sub_dir_list, trial_name="test"):
		# create trial directory
		self.trial_dir = self.__create_trial_dir(trial_name)
		self.sub_dir_dict = self.__create_sub_dirs(sub_dir_list)

		# backup source code
		key_in = [".py", ".m", ".sh", ".txt", "Dockerfile"]
		key_out = [".pyc", ".png", ".npy", "__pycache__", '.npz', '.pkl', '.zip', '.mat']
		backup_keywords(self.trial_dir / "src.zip", key_in, key_out, src_loc=".")


	def __create_trial_dir(self, trial_name):
		base_dir = Path("./trials")
		base_dir.mkdir(exist_ok=True)

		dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"__{trial_name}"
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