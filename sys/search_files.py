import re
import glob
from pathlib import Path
from typing import Pattern, Union, List
import pandas as pd


def escape_special_chars(pattern: str) -> str:
	"""
	괄호 등 re에서 의미를 갖는 특수문자를 이스케이프 처리합니다.
	"""
	special_chars = r'()'
	for c in special_chars:
		pattern = pattern.replace(c, rf'\{c}')
	return pattern


def search_items(root_dir: str,
                 pattern: Union[str, Pattern],
                 search_type: str = 'both') -> List[Path]:
	"""
	root_dir 아래에서 pattern과 매치되는 파일/폴더를 검색하여 list[Path] 형태로 반환합니다.
	(디렉토리나 파일의 이름(name)만을 기준으로 패턴 매칭)

	Parameters
	----------
	root_dir : str
		검색할 루트 경로
	pattern : Union[str, Pattern]
		검색할 정규표현식 패턴(문자열 or re.compile 객체)
	search_type : str
		'files', 'directories' 또는 'both' 중 선택
		- 'files': 파일만
		- 'directories': 디렉토리만
		- 'both': 둘 다

	Returns
	-------
	List[Path]
		pattern이 매칭된 파일/디렉토리 Path 객체 리스트
	"""
	# 만약 문자열 패턴이면, 특수문자 이스케이프 후 compile
	if isinstance(pattern, str):
		escaped_pattern = escape_special_chars(pattern)
		pattern = re.compile(escaped_pattern)

	# 모든 하위 아이템(파일/폴더) 수집
	all_paths = [Path(p) for p in glob.glob(str(Path(root_dir) / '**'), recursive=True)]

	# 검색 타입에 따라 1차 필터링
	if search_type == 'files':
		candidates = [p for p in all_paths if p.is_file()]
	elif search_type == 'directories':
		candidates = [p for p in all_paths if p.is_dir()]
	else:
		candidates = all_paths  # 'both'

	# 최종적으로 이름(p.name)에 대한 정규표현식 매칭
	matched = [p for p in candidates if pattern.search(p.name)]
	return matched


def make_dataframe(paths: List[Path]) -> pd.DataFrame:
	"""
	List[Path]를 받아서 DataFrame으로 만들어줍니다.

	Parameters
	----------
	paths : List[Path]
		파일/디렉토리 Path 객체 리스트

	Returns
	-------
	pd.DataFrame
		Path, Parent, Name, Type, Extension 컬럼을 가진 DataFrame
	"""
	data = {
		"Path": paths,
		"Parent": [p.parent for p in paths],
		"Name": [p.name for p in paths],
		"Type": ["File" if p.is_file() else "Directory" for p in paths],
		"Extension": [p.suffix if p.is_file() else "" for p in paths],
	}
	return pd.DataFrame(data)


def search_items_df(root_dir: str,
                    pattern: Union[str, Pattern],
                    search_type: str = 'both') -> pd.DataFrame:
	"""
	search_items와 make_dataframe를 순차적으로 호출하여,
	매칭된 파일/폴더를 DataFrame으로 반환하는 편의 함수.
	"""
	matched_paths = search_items(root_dir, pattern, search_type)
	return make_dataframe(matched_paths)


if __name__ == "__main__":
	# 사용 예시

	"""
	1) txt 파일 검색 후 DataFrame
		re 는 파일 이름만 검색. 
		- good: r"chunk_.*\.npz" 
		- bad: r".*/chunk_.*\.npz"
	"""
	root_directory = "/path/to/root/directory"
	search_pattern = r".*\.txt$"  # 정규표현식
	result_df = search_items_df(root_directory, search_pattern, search_type="files")
	print(result_df)

	# 2) 특정 문자열을 포함하는 폴더 검색 후 DataFrame
	root_directory = "/path/to/root/directory"
	search_pattern = "first_trial"  # 단순 문자열도 가능
	result_df = search_items_df(root_directory, search_pattern, search_type="directories")
	print(result_df)
