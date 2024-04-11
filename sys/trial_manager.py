
import os
from datetime import datetime

from util_sac.sys.zip_keys import backup_keywords


def create_folder(folder_path, if_exist_rule="error"):

	"""
	Create Folder
		if_exist_rule: "error" -> raise error
					   "ignore" -> print message
	"""
	# create folder if not exist
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
		print(f"folder {folder_path} created")
	else:
		# arise error
		if if_exist_rule == "error":
			raise ValueError(f"folder {folder_path} already exists")
		elif if_exist_rule == "ignore":
			print(f"folder {folder_path} already exists")
		else:
			raise ValueError(f"if_exist_rule {if_exist_rule} not recognized")

	return folder_path



def bus_src(fn_zip, src_loc="."):

	"""
	assumes dir_backup is already exists

	1. find files having some keywords in the file name
	2. exclude files having some keywords in the file name
	3. zip them and copy it to ./backup
	"""

	# set keywords
	key_must = []
	key_in = [".py", ".sh", ".txt"]
	key_out = [".pyc", ".png", "res", ".npy", "__pycache__"]

	# zip file
	backup_keywords(fn_zip, key_must, key_in, key_out, src_loc=src_loc)




class trial_manager:
	"""
	trial_manager

	Rule:
		1. result 가 저장될 drive folder 를 자동으로 생성
		2. backup source code
		3. trial 의 parameter 를 저장
		4. trial_folder 안에 mkdir 과, 생성된 dir 의 name:path pair 를 저장하는 method 를 제공

	Stored Params:
		self.trial_name = "trial_1"
		self.trial_description = "some description about trial_1"

		self.trial_dir = "results/2022-12-01_18-18-19_trial_1"
		self.sub_dirs = {name: path, ...} of created sub dirs


	Useage:
		tm = trial_manager("trial_1", "some description about trial_1")
		pass [tm.res_dir] to the model
		or tm.create_subdir("Xi") -> tm.sub_dirs["Xi"]
	"""

	def __init__(self, trial_name, trial_description):

		"""
		1. Generate file structure
		2. Backup source code
		"""

		"""
		Generate the file structure
		"""

		# store params
		self.trial_name = trial_name
		self.trial_description = trial_description

		# check if "res" folder exists
		self.res_dir = create_folder('results', if_exist_rule='ignore')

		# trial_start_time in yyyy-mm-dd_hh-mm-ss format
		self.trial_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

		# trial folder name = date + trial_name
		self.trial_folder_name = self.trial_start_time + '_' + self.trial_name
		"""
		e.g. 2020-01-01_00-00-00_trial_1
		"""

		# create trial folder
		self.trial_dir = create_folder(os.path.join(self.res_dir, self.trial_folder_name))


		"""
		backup source code
		"""
		backup_fn = os.path.join(self.trial_dir, f"src_{self.trial_description}.zip")
		bus_src(backup_fn , src_loc="..")


		"""
		store trial parameters
		"""
		self.sub_dirs = {}


	def create_subdir(self, dir_name):

		"""
		1. create dir_name in self.trial_dir
		2. return dir_name, dir_path
		"""

		dir_path = create_folder(os.path.join(self.trial_dir, dir_name))
		self.sub_dirs[dir_name] = dir_path

