
import subprocess

import pandas as pd


def get_wmctrl_output_as_df():
	"""
	`wmctrl -l` 명령어의 출력을 파싱하여 pandas DataFrame으로 반환합니다.
	각 행은 시스템에서 실행 중인 창 하나를 나타내며, 열은 창 ID, 데스크탑 번호, 호스트 이름, 창 제목을 포함합니다.
	"""
	# `wmctrl -l` 명령어 실행
	process = subprocess.Popen(['wmctrl', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout, stderr = process.communicate()

	# 출력을 디코딩하고 각 라인으로 분리
	output_lines = stdout.decode('utf-8').strip().split('\n')

	# 출력 파싱
	data = []
	for line in output_lines:
		parts = line.split(maxsplit=3)
		if len(parts) == 4:
			data.append(parts)
		else:
			# 예상치 못한 형식의 라인 처리
			print(f"Unexpected line format: {line}")

	# DataFrame 생성
	df = pd.DataFrame(data, columns=['Window ID', 'Desktop No', 'Host Name', 'Window Title'])

	return df
