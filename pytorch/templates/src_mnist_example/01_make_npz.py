
# -*- coding: utf-8 -*-
"""
Created on  Apr 10 2025

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.dict.json_manager import load_json
from util_sac.pytorch.dataloader.data_collector import DataCollector
from util_sac.sys.date_format import add_timestamp_to_string
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette
from util_sac.sys.dir_manager import create_dir
from util_sac.sys.search_files import search_items_df


"""
1) make npz file 
	path       NumPy Array          (81,)                        8.86 KB <U28
	residence  NumPy Array          (81,)                       648.00 B <U2
	gender     NumPy Array          (81,)                       324.00 B <U1
	age        NumPy Array          (81,)                       648.00 B int64
	RealVote   NumPy Array          (81,)                       972.00 B <U3
	error      NumPy Array          (81, 5, 40)                126.56 KB int64
	listAns    NumPy Array          (81, 5, 40)                126.56 KB int64
	pressedKey NumPy Array          (81, 5, 40)                126.56 KB int64
	rt         NumPy Array          (81, 5, 40)                 63.28 KB float32
	StimRaw    NumPy Array          (81, 5, 40)                189.84 KB <U3
	W단어        NumPy Array          (81, 9, 16)                410.06 KB <U9
	W선택단어      NumPy Array          (81, 9, 1)                  25.63 KB <U9
	W응답시간      NumPy Array          (81, 9, 1)                   2.85 KB float32
	W제시자극      NumPy Array          (81, 9, 1)                  22.78 KB <U8
	WDwelling  NumPy Array          (81, 9, 16)                 45.56 KB float32
	WEvalWord  NumPy Array          (81, 9, 16)                 91.12 KB <U2
	WVisits    NumPy Array          (81, 9, 16)                 45.56 KB float32

2) exclude label = ["수신거부", "응답거부", "투표못함"]
"""


def main():
	"""
	"/home/sac/Dropbox/Projects/2022_snow_man/IAT/data" 를 './data' 로 복사.
	"""

	# search for files
	exp_name = "201702"
	source_dir = f"./data/{exp_name}"
	df = search_items_df(source_dir, "info.json")
	print_partial_markdown(df)


	# create new directory
	target_dir = f"{source_dir}_meta"
	create_dir(target_dir)


	# 불러올 CSV 파일 목록
	list_exclude = ["수신거부", "응답거부", "투표못함"]
	csv_files = [
		"error.csv",
		"listAns.csv",
		"pressedKey.csv",
		"rt.csv",
		"StimRaw.csv",
		"W단어.csv",
		"W선택단어.csv",
		"W응답시간.csv",
		"W제시자극.csv",
		"WDwelling.csv",
		"WEvalWord.csv",
		"WVisits.csv"
	]

	# CSV 파일을 읽어올 때 사용할 데이터 구조 정의
	structure = {
		"path": "str",
		"residence": "str",
		"gender": "str",
		"age": "int64",
		"RealVote": "str",
		"error": "int64",
		"listAns": "int64",
		"pressedKey": "int64",
		"rt": "float32",
		"StimRaw": "str",
		"W단어": "str",
		"W선택단어": "str",
		"W응답시간": "float32",
		"W제시자극": "str",
		"WDwelling": "float32",
		"WEvalWord": "str",
		"WVisits": "float32"
	}
	dc = DataCollector(structure)


	# for all files
	for i, row in df.iterrows():

		# print progress
		file_path = row["Path"]
		print(f"Processing {file_path}...")

		# info.json 로드
		data = load_json(str(file_path))

		# 투표결과가 없는 사람을 제외
		if data["RealVote"] in list_exclude:
			continue

		# file path 추가
		data["path"] = str(file_path)

		# info.json이 있는 디렉토리
		json_dir = file_path.parent

		# 미리 정의한 CSV 파일들을 모두 검사한다
		for csv_file in csv_files:
			csv_path = json_dir / csv_file
			if not csv_path.exists():
				# 존재하지 않을 경우 에러를 발생
				raise FileNotFoundError(f"'{csv_path}' 파일이 존재하지 않는다.")

			# CSV를 pandas로 읽어 numpy 배열로 변환
			df_csv = pd.read_csv(csv_path, index_col=0)
			data[csv_file.replace(".csv", "")] = df_csv.to_numpy()


		dc.add_sample(data)
	data = dc.to_numpy()

	# 데이터를 npz 파일로 저장
	print_array_info(data)
	dc.save_npz(target_dir)






if __name__ == "__main__":
	main()
