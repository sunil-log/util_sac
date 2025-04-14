



import shutil
from pathlib import Path

from util_sac.sys.search_files import search_items_df

"""
이 모듈은 지정된 source directory 내에서 특정 search pattern과 일치하는 파일들을 검색하여,
source directory 내에서의 상대적인 경로 구조를 유지하면서 target directory로 복사하는 기능을 제공한다.

주요 기능:
- `copy_files_with_structure`: source directory에서 pattern에 맞는 파일을 찾아 target directory로 구조를 유지하며 복사하는 메인 함수이다.
- `copy_path_simple`: 단일 파일을 target 경로로 복사하는 보조 함수이다. target의 상위 directory가 없으면 생성한다. 파일 metadata는 복사하지 않는다.

의존성:
- pathlib: 파일 시스템 경로를 객체 지향적으로 다루기 위해 사용된다.
- shutil: 고수준 파일 연산(복사 등)을 위해 사용된다 (`copy_path_simple` 내에서 사용).
- util_sac.sys.search_files.search_items_df: source directory 내에서 파일을 검색하는 데 사용된다.

용례:
	아래 예시는 '/home/sac/Dropbox/Projects/RWA (rem without atonia)/RAW data archive' directory 및 그 하위 directory에서
	이름이 'event.csv'인 모든 파일을 찾아 '/home/sac/Downloads' directory 아래에 원래의 directory 구조를 유지하며 복사한다.
	예를 들어, '/home/sac/Dropbox/Projects/RWA (rem without atonia)/RAW data archive/subject1/visit1/event.csv' 파일은
	'/home/sac/Downloads/subject1/visit1/event.csv'로 복사된다.

	```python
	copy_files_with_structure(
		source_dir="/home/sac/Dropbox/Projects/RWA (rem without atonia)/RAW data archive",
		target_dir="/home/sac/Downloads",
		search_pattern="event.csv",
	)
	```
"""




def copy_path_simple(source_path, target_path):
	"""
	source_path(파일)를 target_path 위치로 복사한다.
	target directory가 없으면 생성한다. 파일 metadata는 복사하지 않는다.

	Args:
		source_path (str 또는 Path): 복사할 원본 파일의 경로.
		target_path (str 또는 Path): 파일이 복사될 목적지 경로 (파일 이름 포함).

	Raises:
		ValueError: source_path가 유효한 파일이 아닐 경우 발생한다.
		Exception: 파일 복사 중 다른 예외 발생 시 전파될 수 있다.
	"""
	source = Path(source_path)
	target = Path(target_path)

	# target의 부모 디렉토리가 존재하지 않으면 생성
	target.parent.mkdir(parents=True, exist_ok=True)

	if source.is_file():
		# metadata 없이 파일 내용만 복사
		shutil.copy(source, target)
	else:
		raise ValueError(f"Invalid source path: {source_path}")


def copy_files_with_structure(
		source_dir,
		target_dir,
		search_pattern
):
	"""
	source directory 내에서 search pattern과 일치하는 파일들을 검색하여,
	source directory 기준의 상대 경로 구조를 유지하며 target directory로 복사한다.

	Args:
		source_dir (str 또는 Path): 검색을 시작할 원본 directory 경로.
		target_dir (str 또는 Path): 파일들을 복사할 목적지 directory 경로.
		search_pattern (str): 검색할 파일 패턴 (예: '*.txt', 'event.csv'). `search_items_df`에서 지원하는 패턴을 사용한다.

	Returns:
		None. 복사 과정 중 발생하는 정보나 오류는 표준 출력으로 print된다.
	"""
	source = Path(source_dir).resolve() # 경로 정규화
	target = Path(target_dir).resolve() # 경로 정규화

	# source_dir 자체가 존재하지 않으면 에러 처리 또는 생성 로직 필요
	if not source.is_dir():
		print(f"Error: Source directory '{source}' not found or is not a directory.")
		return

	# target_dir 생성 (필요한 경우)
	target.mkdir(parents=True, exist_ok=True)

	try:
		# search_items_df 가 Path 객체를 반환한다고 가정
		df = search_items_df(source, search_pattern, search_type="files")
	except Exception as e:
		print(f"Error searching files: {e}")
		return

	if df.empty:
		print("No files found matching the pattern.")
		return

	# copy
	for i, row in df.iterrows():
		source_file_path = Path(row['Path']) # row['Path']가 문자열일 경우 Path 객체로 변환

		# source_dir 기준 상대 경로 계산
		try:
			relative_path = source_file_path.relative_to(source)
		except ValueError:
			# source_file_path가 source_dir 하위에 없는 경우 (로직 오류 또는 예외 상황)
			print(f"Warning: File '{source_file_path}' seems not under source directory '{source}'. Skipping.")
			continue

		# target 경로 생성 (source의 상대 경로 유지)
		target_file_path = target / relative_path

		print(f"Copying {i+1}/{len(df)}: {source_file_path} to {target_file_path}")
		try:
			# copy_path_simple 호출 시 target_file_path 전달
			copy_path_simple(source_file_path, target_file_path)
		except Exception as e:
			print(f"Error copying {source_file_path}: {e}")




if __name__ == '__main__':

	"""
	copy_path 가 그냥 파일 하나만 복사하는거라서,
	search 랑 combine 된 function 이 필요할듯.
	"""


	"""
	# data list = all h5 subject files (load_subject can handle file_path)
	df_data = search_files_by_pattern("/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원", pattern=r".*/raw_microvolt\.npz$")

	# target dir
	dir_remove = "/media/sac/WD4T/Projects_backup/eeg_data/RBD"
	dir_append = "/home/sac/RBD_data"
	df_data['target'] = df_data['File Path'].apply(lambda x: str(x).replace(dir_remove, dir_append))

	# copy
	for i, row in df_data.iterrows():
		print(f"Copying {i+1}/{len(df_data)}: {row['File Path']} to {row['target']}")
		# copy_path(row['File Path'], row['target'])

	"""
	pass


