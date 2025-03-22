

import shutil
from pathlib import Path
from typing import Union


def check_exists(path: Union[str, Path]) -> bool:
	"""경로가 존재하는지 확인한다."""
	return Path(path).exists()


def remove_dir(directory: Union[str, Path]) -> None:
	"""디렉토리를 제거한다. 단, 존재하지 않을 경우 FileNotFoundError를 발생시킨다."""
	try:
		if check_exists(directory):
			print(f"> Removing directory: {directory}")
			shutil.rmtree(directory)
		else:
			raise FileNotFoundError(f"{directory} does not exist")
	except FileNotFoundError as e:
		print(f"> Removal error: {e}")


def create_dir(directory: Union[str, Path]) -> None:
	"""디렉토리를 생성한다. 이미 존재하면 아무것도 하지 않는다."""
	try:
		if not check_exists(directory):
			print(f"> Creating directory: {directory}")
			Path(directory).mkdir(parents=True, exist_ok=True)
		else:
			print(f"> {directory} already exists - do nothing")
	except OSError as e:
		print(f"> Creation error: {e}")


def renew_dir(directory: Union[str, Path]) -> None:
	"""디렉토리가 존재하면 제거 후, 새로 생성한다."""
	remove_dir(directory)
	create_dir(directory)


if __name__ == "__main__":
	directory = Path("path/to/directory")

	# 디렉토리가 존재하는지 확인한다.
	if check_exists(directory):
		print(f"> {directory} exists")
	else:
		print(f"> {directory} does not exist")

	# 디렉토리를 생성한다.
	create_dir(directory)

	# 디렉토리를 제거한다.
	remove_dir(directory)

	# 디렉토리를 갱신(재생성)한다.
	renew_dir(directory)