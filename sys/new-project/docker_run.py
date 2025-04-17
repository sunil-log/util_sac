#!/usr/bin/env python3
import os
import subprocess
import sys


def find_valid_dir(list_dir_candidates):
	"""
	주어진 디렉터리 후보 목록(list_dir_candidates) 중
	가장 먼저 존재하는 디렉터리를 반환한다.
	존재하지 않으면 스크립트를 종료한다.
	"""
	for path in list_dir_candidates:
		if os.path.isdir(path):
			return path

	print("유효한 디렉터리를 찾을 수 없습니다.")
	sys.exit(1)

def run_docker(**kwargs):
	"""
	Docker 컨테이너를 실행한다.
	in_entry: 고정값 "sac"
	필요한 변수는 모두 **kwargs로 받으며, 
	예) user_name, dir_data, out_entry 등

	사용 예시:
	run_docker(
		user_name="sac",
		dir_data="/path/to/trials",
		out_entry="/home/sac/projects"
	)
	"""
	in_entry = "sac"  # 고정값

	user_name = kwargs.get("user_name", "")
	dir_data = kwargs.get("dir_data", "")
	out_entry = kwargs.get("out_entry", "")

	docker_cmd = ["docker", "run"]

	# user 이름이 sac일 때만 -u $(id -u):$(id -g) 옵션 추가
	if user_name == "sac":
		docker_cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"])

	# 볼륨, 기타 Docker 옵션 추가
	docker_cmd.extend([
		"-v", f"{out_entry}:/{in_entry}",
		"-v", f"{dir_data}:/data_dir",
		"-v", "/usr/share/fonts:/usr/share/fonts:ro",
		"--entrypoint", f"/{in_entry}/docker_entrypoint.sh",
		"--rm",
		"--gpus", "all",
		"sac/lightning"
	])

	subprocess.run(docker_cmd, check=True)

def main():
	# main에서 자유롭게 변수만 수정

	list_data_candidates = [
		"/home/sac/RBD_data",
		"/mnt/sdb/RBD_data"
	]
	dir_data = find_valid_dir(list_data_candidates)

	user_name = os.getlogin()
	out_entry = os.getcwd()

	# 실행
	run_docker(
		user_name=user_name,
		dir_data=dir_data,
		out_entry=out_entry
	)

if __name__ == "__main__":
	main()

