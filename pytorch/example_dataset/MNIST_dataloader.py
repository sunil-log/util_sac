
import torch
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


def load_dataloader(batch_size=64, train=True, flatten=True, cnn=True, shuffle=True):
	images, targets = prepare_data(train, flatten, cnn)
	dataset = MNIST_dataset(images, targets)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

	return dataloader



def prepare_data_combine_train_test(flatten=True, cnn=True):
	"""
	MNIST의 train과 test 데이터를 모두 로드한 후 합쳐서 반환한다.
	모든 이미지 픽셀을 0-255에서 0-1 범위로 정규화한다.

	수행 작업:
	1. MNIST train, test 데이터를 모두 다운로드하고 로드한다.
	2. 두 데이터를 합친 뒤 이미지 픽셀 값을 0-1 범위로 변환한다.
	3. 사용자가 원하면 이미지를 1차원으로 평탄화하거나(CNN이 아니라면) 채널 차원을 추가한다(CNN일 경우).
	4. 합쳐진 이미지와 라벨을 dict 형태로 반환한다.

	매개변수:
	flatten (bool): 이미지를 1차원으로 평탄화할지 여부이다 (기본값: True).
	cnn (bool): CNN용 차원을 추가할지 여부이다 (기본값: True).

	반환값:
	dict:
		{
			"x": 합쳐진 MNIST 이미지 (torch.Tensor),
			"y": 해당하는 라벨 (torch.Tensor)
		}
	"""

	# MNIST train과 test 데이터를 로드한다.
	train_data = torchvision.datasets.MNIST(root='./data',
											train=True,
											download=True)
	test_data = torchvision.datasets.MNIST(root='./data',
										   train=False,
										   download=True)

	# train과 test 이미지를 합친다.
	images = torch.cat([
		train_data.data.float() / 255.0,
		test_data.data.float() / 255.0
	], dim=0)

	# train과 test 라벨을 합친다.
	targets = torch.cat([
		train_data.targets,
		test_data.targets
	], dim=0)

	# flatten 옵션에 따라 이미지를 1차원으로 평탄화한다.
	if flatten:
		images = images.view(-1, 28 * 28)

	# cnn 옵션에 따라 채널 차원을 추가한다.
	if cnn:
		images = images.unsqueeze(1)

	return {"x": images, "y": targets}


if __name__ == "__main__":
	# train용 DataLoader
	train_loader = load_dataloader(batch_size=64, train=True, flatten=True, cnn=True, shuffle=True)
	test_loader = load_dataloader(batch_size=64, train=False, flatten=True, cnn=True, shuffle=False)

	# 간단한 테스트 코드 (train)
	for batch_idx, (images, targets) in enumerate(train_loader):
		print(f"[Train] Batch {batch_idx} - images.shape: {images.shape}, targets.shape: {targets.shape}")
		# 필요한 로직을 여기에 추가
		break

	# 간단한 테스트 코드 (test)
	for batch_idx, (images, targets) in enumerate(test_loader):
		print(f"[Test] Batch {batch_idx} - images.shape: {images.shape}, targets.shape: {targets.shape}")
		# 필요한 로직을 여기에 추가
		break
