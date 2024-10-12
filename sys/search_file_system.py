import re
from pathlib import Path
import glob
from typing import Pattern, List, Union
import pandas as pd

class FileAndDirectorySearcher:
	def __init__(self, root_dir: str):
		self.root_dir = Path(root_dir)

	def search(self, pattern: Pattern, search_type: str = 'both') -> pd.DataFrame:
		try:
			paths = self._find_items(pattern, search_type)
			return self._create_dataframe(paths, search_type)
		except Exception as e:
			raise SearchError(f"Error occurred while searching: {e}")

	def _find_items(self, pattern: Pattern, search_type: str) -> List[Path]:
		try:
			items = [Path(item) for item in glob.glob(str(self.root_dir / "**"), recursive=True) if pattern.search(str(item))]
			if search_type == 'files':
				return [item for item in items if item.is_file()]
			elif search_type == 'directories':
				return [item for item in items if item.is_dir()]
			else:  # 'both'
				return items
		except Exception as e:
			raise SearchError(f"Error occurred while finding items: {e}")

	def _create_dataframe(self, paths: List[Path], search_type: str) -> pd.DataFrame:
		try:
			data = {
				"Path": paths,
				"Parent": [path.parent for path in paths],
				"Name": [path.name for path in paths],
				"Type": ["File" if path.is_file() else "Directory" for path in paths]
			}
			if search_type == 'files' or search_type == 'both':
				data["Extension"] = [path.suffix for path in paths]
			return pd.DataFrame(data)
		except Exception as e:
			raise SearchError(f"Error occurred while creating DataFrame: {e}")

class SearchError(Exception):
	pass

def escape_special_chars(pattern: str) -> str:
	special_chars = r'()'
	return ''.join([f'\\{c}' if c in special_chars else c for c in pattern])

def search_by_pattern(root_dir: str, pattern: Union[str, Pattern], search_type: str = 'both') -> pd.DataFrame:
	try:
		if isinstance(pattern, str):
			escaped_pattern = escape_special_chars(pattern)
			pattern = re.compile(escaped_pattern)
		searcher = FileAndDirectorySearcher(root_dir)
		return searcher.search(pattern, search_type)
	except SearchError as e:
		print(f"SearchError: {e}")
		return pd.DataFrame()



if __name__ == '__main__':


	"""
	file_search.py 에서 dir 도 찾을 수 있게 만든 버젼
	"""


	"""
	모든 txt 파일을 찾아서 DataFrame으로 반환하는 예시
	"""
	root_directory = "/path/to/root/directory"
	search_pattern = r".*\.txt$"  # 파일 확장자가 .txt인 파일 검색
	result_df = search_by_pattern(root_directory, search_pattern, search_type='files')

	"""
	'first_trial' 을 포함하는 모든 디렉토리를 찾아서 DataFrame으로 반환하는 예시
	"""
	root_directory = "/path/to/root/directory"
	search_pattern = "first_trial"
	result_df = search_by_pattern(root_directory, search_pattern, search_type='directories')



