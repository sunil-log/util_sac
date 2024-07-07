

import torchvision
from torch.utils.data import DataLoader, Dataset

"""
MNIST class 는 dataloader 에 넣을 때 transform 을 한번에 적용하는 것이 아니라,
dataloader 에 의해서 호출될 때마다 transform 을 적용한다. 그래서 느려진다.
차라리 data 를 transform 한 뒤 torch tensor 로 변환하여 dataloader 에 넣는게 낫다.
"""


class MNIST_dataset(Dataset):
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.targets[idx]


def prepare_data(batch_size=64, train=True, flatten=True, cnn=True):
	"""
	MNIST 데이터셋을 로드하고 전처리하여 DataLoader 객체로 반환합니다.
	정규화 과정에서 모든 픽셀 값을 255로 나누어 0-1 범위로 변환합니다.

	이 함수는 다음과 같은 작업을 수행합니다:
	1. MNIST 데이터셋을 다운로드합니다 (필요한 경우).
	2. 이미지 픽셀 값을 0-255에서 0-1 범위로 정규화합니다.
	3. 선택적으로 이미지를 1차원으로 평탄화하거나 CNN용 차원을 추가합니다.
	4. DataLoader 객체를 생성하여 반환합니다.

	매개변수:
	batch_size (int): 배치 크기 (기본값: 64)
	train (bool): 학습 데이터셋 사용 여부 (기본값: True)
	flatten (bool): 이미지를 1차원으로 평탄화할지 여부 (기본값: True)
	cnn (bool): CNN용 차원을 추가할지 여부 (기본값: True)

	반환값:
	DataLoader: 전처리된 MNIST 데이터를 포함하는 DataLoader 객체
	"""
	if train:
		data = torchvision.datasets.MNIST(root='./data',
		                                  train=True,
		                                  download=True)
	else:
		data = torchvision.datasets.MNIST(root='./data',
		                                  train=False,
		                                  download=True)

	images = data.data.float() / 255.0
	targets = data.targets

	if flatten:
		images = images.view(-1, 28 * 28)

	if cnn:
		images = images.unsqueeze(1)

	dataset = MNIST_dataset(images, targets)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return dataloader