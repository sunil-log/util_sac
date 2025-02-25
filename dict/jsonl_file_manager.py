```python
"""
이 모듈은 Line-based JSON 파일(.jsonl)을 효율적으로 관리하기 위해 설계되었습니다.
각 줄에 하나의 JSON 형태 데이터를 저장하고, 이를 읽어오는 과정에서 dict 타입의 리스트를 얻을 수 있습니다.
또한 이 데이터를 pandas의 DataFrame 형태로 변환할 수도 있습니다.

용례:
	from my_module import jsonl_file_manager
	from pathlib import Path

	# 파일 매니저 인스턴스 생성 (str 또는 Path 사용 가능)
	manager = JsonlFileManager(filepath=Path("data.jsonl"))

	# dict 데이터를 한 줄씩 저장
	manager.write_line({"id": 1, "text": "Hello World"})
	manager.write_line({"id": 2, "text": "Another line"})

	# 파일에서 dict 목록 읽어오기
	data_list = manager.read()
	print(data_list)
	# [{'id': 1, 'text': 'Hello World'}, {'id': 2, 'text': 'Another line'}]

	# DataFrame으로 변환
	df = manager.read_as_df()
	print(df)
	#	 id			 text
	# 0	1	 Hello World
	# 1	2  Another line

주의사항:
	- 파일 경로로 str 혹은 Path 모두 사용 가능합니다.
	- JSON 형식이 아닌 라인이 존재할 경우, 해당 라인은 무시됩니다.
	- 확장자는 일반적으로 .jsonl(또는 .ndjson)을 권장하지만, .txt로도 동작에 문제는 없습니다.
"""

from pathlib import Path
from typing import Union
import json
import pandas as pd

class jsonl_file_manager:
	"""
	Line-based JSON 파일(.jsonl)을 관리하는 클래스입니다.
	- __init__(filepath): 파일 경로를 받아 self.filepath에 저장합니다.
	  (str 또는 Path 객체 모두 가능)
	- write_line(data): dict를 받아 JSON 형식으로 한 줄씩 파일에 저장합니다.
	- read(): 파일의 모든 줄을 읽고, 각 줄을 dict로 변환한 뒤 list로 반환합니다.
	- read_as_df(): read()로 얻은 dict list를 DataFrame으로 변환하여 반환합니다.
	"""
	def __init__(self, filepath: Union[str, Path]):
		"""
		filepath(str or Path): 파일(.jsonl 등)의 경로
		"""
		self.filepath = str(filepath)

	def write_line(self, data):
		"""
		data(dict): 파일에 JSON 형식으로 한 줄 저장할 데이터
		"""
		with open(self.filepath, 'a', encoding='utf-8') as f:
			json_string = json.dumps(data, ensure_ascii=False)
			f.write(json_string + '\n')

	def read(self):
		"""
		파일의 모든 줄을 읽고, 각 줄을 dict로 변환하여 list로 반환합니다.
		변환이 불가능한 줄은 무시합니다.

		Returns:
			list of dict: 파일에서 읽어온 모든 데이터를 담고 있는 리스트
		"""
		data_list = []
		with open(self.filepath, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					data_list.append(json.loads(line))
				except json.JSONDecodeError:
					# JSON 형태가 아니면 무시
					pass
		return data_list

	def read_as_df(self):
		"""
		read()로 읽은 dict list를 기반으로 DataFrame을 생성하여 반환합니다.

		Returns:
			pandas.DataFrame: 파일에서 읽어온 데이터를 담고 있는 DataFrame
		"""
		data_list = self.read()
		return pd.DataFrame(data_list)
```