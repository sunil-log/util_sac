
import torch

"""
see: module, TSRep, dataloader, part 01, extract window
"""

def batch_windows(x, indices, window_size=10, future=False):
	"""
	- x: 전체 시계열 데이터 (shape: [channel, sequence_length])
	- indices: 추출하고자 하는 윈도우의 끝 인덱스들 (shape: [batch_size])
	- window_size: 각 윈도우의 크기

	결과물
	- shape이 [batch_size, channel, window_size]인 tensor

	비고
    - 이 함수는 과거와 미래 윈도우 모두를 추출할 수 있습니다.
    - GPU에서 효율적으로 동작하도록 설계되었습니다.
	"""

	# 배치 크기 계산
	batch_size = len(indices)

	"""
	인덱스 텐서 생성
	"""
	idx = torch.arange(window_size, device=x.device).unsqueeze(0).expand(batch_size, -1)
	"""
	idx.shape = (3, 10)
	tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device='cuda:0')
	"""

	"""
	주어진 인덱스=(300, 400, 500)를 기준으로 윈도우의 시작 인덱스를 계산
	"""
	if future:
		idx = indices.unsqueeze(1) + idx
	else:
		idx = indices.unsqueeze(1) - window_size + idx
	"""
	if past:
	tensor([[290, 291, 292, 293, 294, 295, 296, 297, 298, 299],
            [390, 391, 392, 393, 394, 395, 396, 397, 398, 399],
            [490, 491, 492, 493, 494, 495, 496, 497, 498, 499]], device='cuda:0')
	"""

	# check if index has negative value
	if torch.any(idx < 0):
		raise ValueError("Negative index is not allowed")


	"""
	윈도우 추출
	"""
	windows = x[:, idx]
	"""
	x.shape = (channel=2, sequence_length=5000)
	idx.shape = (batch_size=3, window_size=10)
	
	windows.shape = torch.Size([channel=2, batch_size=3, window_length=10])
	tensor([[[ 290.,  291.,  292.,  293.,  294.,  295.,  296.,  297.,  298.,  299.],
	         [ 390.,  391.,  392.,  393.,  394.,  395.,  396.,  397.,  398.,  399.],
	         [ 490.,  491.,  492.,  493.,  494.,  495.,  496.,  497.,  498.,  499.]],
	
	        [[1290., 1291., 1292., 1293., 1294., 1295., 1296., 1297., 1298., 1299.],
	         [1390., 1391., 1392., 1393., 1394., 1395., 1396., 1397., 1398., 1399.],
	         [1490., 1491., 1492., 1493., 1494., 1495., 1496., 1497., 1498., 1499.]]],
	       device='cuda:0')
	"""

	# 차원 순서 변경: [channel, batch, window_size] -> [batch, channel, window_size]
	windows = windows.permute(1, 0, 2)
	"""
	windows.shape = torch.Size([batch_size=3, channel=2, window_length=10])
	"""


	return windows



class TimeSeriesIndexLoader():
	def __init__(self,
	             key_idx, idx_interval=1,
	             batch_size=3,
	             shuffle=False):
		"""
		Input
		- x: 전체 시계열 데이터 (shape: [channel, sequence_length])
		- window_size: 각 윈도우의 크기
		- batch_size: 배치 크기

		Example
		- idx_start: 0,
		- idx_end: 100,
		- batch_size: 3,

		Output
		- batch_past_indices:
			[0, 1, 2], [3, 4, 5], ..., [96, 97, 98], [99]
		- if shuffle=True
			[3, 0, 2], [1, 4, 5], ..., [21, 13, 19], [72]

		Note
		- output 을 current time 으로 사용하여, batch_windows 를 수행해면 된다.
		- batch_windows() 를 내부적으로 수행하면 translation jitter 를 적용하지 못하여, index 만 반환
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indices = torch.arange(key_idx[0], key_idx[1], idx_interval)
		self.current_idx = 0

	def __iter__(self):
		self.current_idx = 0
		if self.shuffle:
			self.indices = self.indices[torch.randperm(len(self.indices))]
		return self

	def __next__(self):
		if self.current_idx >= len(self.indices):
			raise StopIteration

		batch_end = min(self.current_idx + self.batch_size, len(self.indices))
		batch_indices = self.indices[self.current_idx:batch_end]

		# update current index
		self.current_idx = batch_end

		return batch_indices

def train_test_index_split(len_x,
                           train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,
                           past_window_size=100, future_window_size=10):

	# check ratio
	assert train_ratio + valid_ratio + test_ratio == 1

	train_start = past_window_size
	train_end = int(len_x * train_ratio)

	valid_start = train_end
	valid_end = int(len_x * (train_ratio + valid_ratio))

	test_start = valid_end
	test_end = len_x - future_window_size

	d = {
		"train": (train_start, train_end),
		"valid": (valid_start, valid_end),
		"test": (test_start, test_end)
	}
	print(f"train: {d['train']}, valid: {d['valid']}, test: {d['test']}", flush=True)

	return d


def sample_data(size=5000):
	x1 = torch.arange(size, dtype=torch.float32)
	x2 = x1 + 1000
	return torch.stack([x1, x2], dim=0) + 0.5



def main():

	# GPU 사용 가능 여부 확인
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# 시계열 데이터 생성
	x = sample_data()
	x = x.to(device)
	"""
	x.shape = (channel=2, ts=5000)
	"""

	# train test key index 생성
	past_window_size = 10
	future_window_size = 3
	key_idx = train_test_index_split(
		len_x=x.shape[1],
		past_window_size=past_window_size,
		future_window_size=future_window_size
	)
	"""
	key_idx = {train: (10, 4000), valid: (4000, 4500), test: (4500, 4997)}
	"""

	train_indexloader = TimeSeriesIndexLoader(
		key_idx=key_idx['train'],
		idx_interval=1, batch_size=3, shuffle=False)


	# loop over batches
	for i, batch in enumerate(train_indexloader):
		indices = batch
		indices = indices.to(device)
		"""
		indices = torch.tensor([0, 1, 2], device='cuda:0')
		indices.shape = (3)
		"""

		"""
		윈도우 추출
		"""
		x_past = batch_windows(x, indices, window_size=past_window_size, future=False)
		x_future = batch_windows(x, indices, window_size=future_window_size, future=True)
		"""
		x_past = 
			[[[   0.5    1.5    2.5    3.5    4.5    5.5    6.5    7.5    8.5    9.5]
			  [1000.5 1001.5 1002.5 1003.5 1004.5 1005.5 1006.5 1007.5 1008.5 1009.5]]
			
			 [[   1.5    2.5    3.5    4.5    5.5    6.5    7.5    8.5    9.5   10.5]
			  [1001.5 1002.5 1003.5 1004.5 1005.5 1006.5 1007.5 1008.5 1009.5 1010.5]]
			
			 [[   2.5    3.5    4.5    5.5    6.5    7.5    8.5    9.5   10.5   11.5]
			  [1002.5 1003.5 1004.5 1005.5 1006.5 1007.5 1008.5 1009.5 1010.5 1011.5]]]

		x_future =
			[[[  10.5   11.5   12.5]
			  [1010.5 1011.5 1012.5]]
			
			 [[  11.5   12.5   13.5]
			  [1011.5 1012.5 1013.5]]
			
			 [[  12.5   13.5   14.5]
			  [1012.5 1013.5 1014.5]]]
		"""

		print(f"Batch {i}:")
		print(f"x_past shape: {x_past.shape}")
		print(f"x_future shape: {x_future.shape}")

		if i >= 2:  # 처음 3개 배치만 출력
			break


if __name__ == '__main__':
	main()