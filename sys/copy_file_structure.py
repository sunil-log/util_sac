

import shutil
from pathlib import Path


def create_directory_recursive(target_dir: str) -> None:
	"""
	target_dir="./the/target/dir/" 이 주어졌을 때, 해당 dir 이 존재하지 않으면 recursuve 하게 dir 을 생성
	"""
	path = Path(target_dir)
	if not path.exists():
		path.mkdir(parents=True)


def copy_path(source_path, target_path):
	"""
	source_path 와 target_path 가 주어져 있는 경우 그것들을 복사를 하는 함수.
	source_path 혹은  target_path 는 str 이나 Path() 일 수 있다.
	"""

	# source_path와 target_path를 Path 객체로 변환
	source = Path(source_path)
	target = Path(target_path)

	# 대상 디렉토리가 존재하지 않으면 생성
	create_directory_recursive(target.parent)

	# 파일인 경우 파일 복사
	if source.is_file():
		shutil.copy(source, target)
	# 디렉토리인 경우 디렉토리 복사
	elif source.is_dir():
		shutil.copytree(source, target)
	else:
		raise ValueError(f"Invalid source path: {source_path}")



if __name__ == '__main__':

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