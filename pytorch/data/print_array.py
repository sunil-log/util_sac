

import numpy as np
import torch


def format_memory_size(memory_bytes):
	"""
	바이트 단위의 크기를 받아 적절한 단위(B, KB, MB, GB, TB)로 변환하여 문자열로 반환합니다.
	"""
	for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
		if memory_bytes < 1024:
			return f"{memory_bytes:.2f} {unit}"
		memory_bytes /= 1024
	return f"{memory_bytes:.2f} PB"  # TB 이상인 경우



def print_array_info_dict(data_dict):
	print(f"{'Key':<10} {'Type':<20} {'Shape':<20} {'Memory':>15} {'Dtype':<10}")
	print("-" * 80)
	for key, data in data_dict.items():
		if isinstance(data, (torch.Tensor, np.ndarray)):
			if isinstance(data, torch.Tensor):
				shape = tuple(data.size())
				memory = data.element_size() * data.nelement()
				dtype = data.dtype
				data_type = "PyTorch Tensor"
			elif isinstance(data, np.ndarray):
				shape = data.shape
				memory = data.nbytes
				dtype = data.dtype
				data_type = "NumPy Array"

			mem_str = format_memory_size(memory)
			print(f"{key:<10} {data_type:<20} {str(shape):<20} {mem_str:>15} {dtype}")
		else:
			print(f"{key:<10} {'Other':<20} {str(type(data)):<20} {'N/A':>15} {'N/A':<10}")


def print_array_info(*args):
	"""
	주어진 args에 대해:
	- 단일 인자인 경우:
	  - dict이면 그대로 print_array_info_dict 함수 호출
	  - NpzFile이면 내부를 unpack하여 dict로 변환 후 print_array_info_dict 함수 호출
	  - 그 외(일반 array 등)는 통상적인 방식으로 dict로 묶어 처리
	- 여러 인자인 경우:
	  - 각 인자를 dict로 묶은 후 print_array_info_dict 함수 호출
	"""

	if len(args) == 1:
		if isinstance(args[0], dict):
			print_array_info_dict(args[0])
		elif isinstance(args[0], np.lib.npyio.NpzFile):
			data_dict = {k: args[0][k] for k in args[0].files}
			print_array_info_dict(data_dict)
		else:
			data_dict = {f"arg_{idx}": data for idx, data in enumerate(args)}
			print_array_info_dict(data_dict)
	else:
		data_dict = {f"arg_{idx}": data for idx, data in enumerate(args)}
		print_array_info_dict(data_dict)

