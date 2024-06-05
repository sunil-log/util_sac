

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