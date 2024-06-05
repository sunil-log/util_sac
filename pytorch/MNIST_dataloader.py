

import torchvision
from torch.utils.data import DataLoader, Dataset


class MNIST_dataset(Dataset):

	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def prepare_data(train=True):
	"""
	MNIST class 는 dataloader 에 넣을 때 transform 을 한번에 적용하는 것이 아니라,
	dataloader 에 의해서 호출될 때마다 transform 을 적용한다. 그래서 느려진다.
	차라리 data 를 transform 한 뒤 torch tensor 로 변환하여 dataloader 에 넣는게 낫다.
	"""

	if train:
		data = torchvision.datasets.MNIST(root='./data',
											train=True,
											download=True)
	else:
		data = torchvision.datasets.MNIST(root='./data',
											train=False,
											download=True)


	# extract data from MNIST dataset
	data = data.data        # torch.Size([60000, 28, 28])


	# 0~255 인 data 를 0~1 사이의 값으로 normalize
	data = data.float() / 255.0
	print(f"Range of data: [{data.min()}, {data.max()}]")

	# data 를 1차원으로 변환
	data = data.view(-1, 28*28)

	# data 를 MNIST_dataset class 로 변환
	dataset = MNIST_dataset(data)

	# data 를 dataloader 로 변환
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

	return dataloader