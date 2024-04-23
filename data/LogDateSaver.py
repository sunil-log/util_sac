import os
import glob
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd


"""
이 class 는 init 시에 root_dir 과 df 를 받아.
dir 이 존재하지 않으면 생성해.
df 는 "date" column 이 있어야 하고 이것은 pd.datetime 이어야 해. 그렇지 않으면 오류가 발생해.

그리고 save_csv method 는 df 를 csv 로 저장하는데, df 를 date 에서 날자(date) 로 group 해서 "./{root_dir}/{date}.csv" 로 저장하여야 해.
그런데 이미 해당 파일이 존재한다면 기존의 파일을 열고 새로운 df 를 append 한 다음 (이 때 기존과 현재 df 의 column 이 일치해야 함) date 로 sort 내림차순으로 하고 csv 로 저장
없다면 그냥 sort 만 하고 저장.

그리고 save_csv 뿐만 아니라 save_h5 도 동일한 기능을 할 수 있어야 한다. 다만 csv 대신 h5 를 저장하는거지.
"""



class LogDateSaver(ABC):
	def __init__(self, root_dir: str, df: pd.DataFrame):
		self.root_dir = Path(root_dir)
		self.root_dir.mkdir(parents=True, exist_ok=True)

		if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
			raise ValueError("Input DataFrame must have a 'date' column of datetime type.")
		self.df = df

	def _process_existing_data(self, file_path: Path, group: pd.DataFrame) -> pd.DataFrame:
		existing_df = self._read_existing_data(file_path)
		if not existing_df.columns.equals(group.columns):
			raise ValueError(f"Columns of existing data at {file_path} do not match the input DataFrame.")
		existing_df["date"] = pd.to_datetime(existing_df["date"])
		combined_df = pd.concat([existing_df, group])
		return combined_df

	@abstractmethod
	def _read_existing_data(self, file_path: Path) -> pd.DataFrame:
		pass

	@abstractmethod
	def _save_data(self, file_path: Path, combined_df: pd.DataFrame):
		pass

	def save(self):
		for date, group in self.df.groupby(self.df["date"].dt.date):
			file_path = self.root_dir / f"{date}.{self._get_extension()}"

			if file_path.exists():
				combined_df = self._process_existing_data(file_path, group)
			else:
				combined_df = group

			combined_df.sort_values("date", ascending=True, inplace=True)
			combined_df.reset_index(drop=True, inplace=True)
			print(combined_df)
			self._save_data(file_path, combined_df)

	@abstractmethod
	def _get_extension(self) -> str:
		pass


class CSVSaver(LogDateSaver):
	def _read_existing_data(self, file_path: Path) -> pd.DataFrame:
		return pd.read_csv(file_path)

	def _save_data(self, file_path: Path, combined_df: pd.DataFrame):
		combined_df.to_csv(file_path, index=False)

	def _get_extension(self) -> str:
		return "csv"


class H5Saver(LogDateSaver):
	def _read_existing_data(self, file_path: Path) -> pd.DataFrame:
		return pd.read_hdf(file_path, "data")

	def _save_data(self, file_path: Path, combined_df: pd.DataFrame):
		combined_df.to_hdf(file_path, key="data", mode="w")

	def _get_extension(self) -> str:
		return "h5"




def save_df_as_log(root_dir: str, df: pd.DataFrame, fmt: str):
	if fmt.lower() == "csv":
		saver = CSVSaver(root_dir, df)
	elif fmt.lower() == "h5":
		saver = H5Saver(root_dir, df)
	else:
		raise ValueError(f"Unsupported format: {fmt}. Supported formats are 'csv' and 'h5'.")

	saver.save()




"""
Example usage:
"""
if __name__ == '__main__':
	import numpy as np
	from datetime import datetime, timedelta

	root_dir = "log"

	# 시작일과 종료일 설정
	start_date = datetime(2024, 1, 1)
	end_date = datetime(2024, 1, 10)

	# 시작일부터 종료일까지의 날짜 범위 생성
	date_range = pd.date_range(start=start_date, end=end_date, freq='S')

	# 임의의 날짜 10개 선택
	n_rows = 100
	random_dates = np.random.choice(date_range, size=n_rows)

	# 임의의 float32 값 10개 생성
	random_values = np.random.rand(n_rows).astype(np.float32)

	# DataFrame 생성
	df = pd.DataFrame({'date': random_dates, 'value': random_values})
	"""
	df = 
		                  date     value
		0  2024-01-01 16:08:56  0.760029
		1  2024-01-07 03:15:33  0.743148
		..                 ...       ...
		98 2024-01-04 07:23:38  0.910436
		99 2024-01-05 21:52:56  0.699800
	"""


	save_df_as_log(root_dir, df, "csv")

