



import shutil
from pathlib import Path

from util_sac.pandas.print_df import print_partial_markdown
from util_sac.sys.search_files import search_items_df


"""

용례
copy_files_with_structure(
	source_dir="/home/sac/Dropbox/Projects/RWA (rem without atonia)/RAW data archive",
	target_dir="/home/sac/Downloads",
	search_pattern="event.csv",
)

"""




def copy_path_simple(source_path, target_path):
	"""
	source_path(파일 또는 디렉토리)를 target_path 위치로 복사한다.
	target 디렉토리가 없으면 생성한다. metadata는 복사하지 않는다.
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


