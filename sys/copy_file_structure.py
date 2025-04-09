



import shutil
from pathlib import Path
from util_sac.sys.dir_manager import create_dir


"""
이 모듈은 디렉토리 및 파일(또는 디렉토리) 복사를 간편하게 처리하기 위한 기능을 제공한다.

주요 함수:
1) copy_path(source_path, target_path) -> None
   - source_path(파일 또는 디렉토리)를 target_path 위치로 복사한다.
   - source_path와 target_path는 str이나 Path() 객체 모두 가능하다.
   - 복사 과정에서 target_path의 부모 디렉토리가 없으면 자동으로 생성한다.
   - 예시:
       copy_path("./source/file.txt", "./the/target/dir/file.txt")
       copy_path("./source/directory", "./the/target/dir/directory_copy")

용례:
- 대규모 directory 구조를 미리 만들어야 할 때 create_directory_recursive를 사용한다.
- 특정 파일 또는 directory를 다른 경로로 옮기거나 백업할 때 copy_path를 사용한다.

주의:
- source_path가 파일인지, directory인지에 따라 복사 동작이 달라진다.
- source_path가 유효하지 않을 경우 ValueError가 발생한다.
"""

"""
다음 Bash code 를 쓰는것이 나을수도 있다.

#!/bin/bash

# 원본 디렉토리와 대상 디렉토리 설정
SOURCE_DIR="/media/sac/WD4T/Projects_backup/eeg_data/RBD/대전성모병원"
TARGET_DIR="/home/sac/RBD_data"

# 파일 찾기 및 복사
find "$SOURCE_DIR" -name "*raw_microvolt.npz" | while read -r file; do
	# 대상 경로 생성
	target_file=$(echo "$file" | sed "s|$SOURCE_DIR|$TARGET_DIR|")

	# 대상 디렉토리 생성
	mkdir -p "$(dirname "$target_file")"

	# 파일 복사
	cp "$file" "$target_file"

	echo "Copied: $file to $target_file"
done
"""



def copy_path(source_path, target_path):
	"""
	source_path(파일 또는 디렉토리)를 target_path 위치로 복사한다.
	source_path와 target_path는 str 또는 Path 객체를 모두 지원한다.

	파일 복사 시 메타데이터까지 복사하기 위해 shutil.copy2 사용을 권장한다.
	디렉토리 복사 시 Python 버전에 따라 shutil.copytree(..., dirs_exist_ok=True) 활용 가능.
	"""
	source = Path(source_path)
	target = Path(target_path)

	# 타깃 디렉토리가 없으면 생성
	create_directory_recursive(target.parent)

	if source.is_file():
		# 메타데이터까지 복사
		shutil.copy2(source, target)
	elif source.is_dir():
		# Python 3.8 이상인 경우
		try:
			shutil.copytree(source, target, dirs_exist_ok=True)
		except TypeError:
			# Python 3.7 이하에서는 dirs_exist_ok 옵션이 없으므로 기본 동작만 수행
			shutil.copytree(source, target)
	else:
		raise ValueError(f"Invalid source path: {source_path}")


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


