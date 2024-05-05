

from pathlib import Path
from typing import Union
import shutil

class dir_manager:
	"""File system management class"""

	def check_exists(self, path: Union[str, Path]) -> bool:
		"""Check if the path exists"""
		if not Path(path).exists():
			return False
		else:
			return True

	def remove_dir(self, directory: Union[str, Path]) -> None:
		"""Remove directory only if it exists"""
		try:
			if self.check_exists(directory):
				print(f"> Removing directory: {directory}")
				shutil.rmtree(directory)
			else:
				raise FileNotFoundError(f"{directory} does not exist")
		except FileNotFoundError as e:
			print(f"> Removal error: {e}")

	def create_dir(self, directory: Union[str, Path]) -> None:
		"""Create directory"""
		try:
			if not self.check_exists(directory):
				print(f"> Creating directory: {directory}")
				Path(directory).mkdir(parents=True, exist_ok=True)
			else:
				print(f"> {directory} already exists - do nothing")
		except OSError as e:
			print(f"> Creation error: {e}")

	def renew_dir(self, directory: Union[str, Path]) -> None:
		"""Renew directory by removing it if it exists and then creating it"""
		self.remove_dir(directory)
		self.create_dir(directory)



# Usage example
if __name__ == "__main__":
	fs_manager = dir_manager()
	directory = Path("path/to/directory")

	# renew directory
	fs_manager.renew_dir(directory)

