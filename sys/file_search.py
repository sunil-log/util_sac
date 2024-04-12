import glob
from pathlib import Path
from typing import List, Pattern
import pandas as pd


class FileSearcher:
	def __init__(self, root_dir: str):
		self.root_dir = Path(root_dir)

	def search_files(self, pattern: Pattern) -> pd.DataFrame:
		try:
			file_paths = self._find_files(pattern)
			return self._create_dataframe(file_paths)
		except Exception as e:
			raise FileSearchError(f"Error occurred while searching files: {e}")

	def _find_files(self, pattern: Pattern) -> List[Path]:
		try:
			return [Path(file) for file in glob.glob(str(self.root_dir / "**" / "*"), recursive=True) if
			        pattern.search(file)]
		except Exception as e:
			raise FileSearchError(f"Error occurred while finding files: {e}")

	def _create_dataframe(self, file_paths: List[Path]) -> pd.DataFrame:
		try:
			data = {"File Path": file_paths}
			return pd.DataFrame(data)
		except Exception as e:
			raise FileSearchError(f"Error occurred while creating DataFrame: {e}")


class FileSearchError(Exception):
	pass


def search_files_by_pattern(root_dir: str, pattern: Pattern) -> pd.DataFrame:
	try:
		searcher = FileSearcher(root_dir)
		return searcher.search_files(pattern)
	except FileSearchError as e:
		print(f"FileSearchError: {e}")
		return pd.DataFrame()



if __name__ == '__main__':

	# Usage example

	import re

	root_directory = "/path/to/root/directory"
	search_pattern = re.compile(r".*\.txt$")  # 파일 확장자가 .txt인 파일 검색

	result_df = search_files_by_pattern(root_directory, search_pattern)
	print(result_df)