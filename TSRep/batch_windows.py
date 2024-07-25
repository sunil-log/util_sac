
import torch

def batch_windows(x, indices, window_size=10):
	"""
	- x: 전체 시계열 데이터 (shape: [channel, sequence_length])
	- indices: 추출하고자 하는 윈도우의 끝 인덱스들 (shape: [batch_size])
	- window_size: 각 윈도우의 크기

	결과물
	- shape이 [batch_size, channel, window_size]인 tensor

	여기서의 index 연산은 GPU 에서 효율적이다.
		따라서 이 함수는 되도록 `x.device=cuda` 일 떄 사용하는 것이 효율적이다.
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
	idx = indices.unsqueeze(1) - window_size + idx
	"""
	tensor([[290, 291, 292, 293, 294, 295, 296, 297, 298, 299],
            [390, 391, 392, 393, 394, 395, 396, 397, 398, 399],
            [490, 491, 492, 493, 494, 495, 496, 497, 498, 499]], device='cuda:0')
	"""

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



def main():

	# GPU 사용 가능 여부 확인
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# 사용 예시
	x1 = torch.arange(5000, dtype=torch.float32)
	x2 = x1 + 1000
	x = torch.stack([x1, x2], dim=0).to(device)  # 전체 시계열 데이터

	indices = torch.tensor([300, 400, 500], device=device)  # 추출하고자 하는 윈도우의 끝 인덱스들
	"""
	x.shape = (channel=2, ts=5000)
	indices.shape = (3)
	"""

	# 윈도우 추출
	windows = batch_windows(x, indices)
	print(f"Windows shape: {windows.shape}")  # torch.Size([




if __name__ == '__main__':
	main()