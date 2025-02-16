import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import os

"""
CIFAR_dataset 클래스는 DataLoader에 넣을 때마다 transform을 반복 적용하는 대신,
미리 이미지를 Tensor로 변환해 놓고 dataloader에 전달합니다.
이 방식은 매 스텝마다 PIL 이미지를 변환하는 과정을 줄여, 속도를 높이는 장점이 있습니다.
"""

class CIFAR_dataset(Dataset):
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.targets[idx]


def load_dataloader(batch_size=64, train=True, flatten=False, shuffle=True):
	"""
	미리 변환된 CIFAR 이미지를 CIFAR_dataset 클래스에 담아 DataLoader로 반환합니다.
	"""
	images, targets = prepare_data_cached(train, flatten)
	dataset = CIFAR_dataset(images, targets)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

	return dataloader



def prepare_data_cached(train=True, flatten=True, cache_path=None):
	"""
	train 인자와 cache_path를 받아서, 캐시 파일이 있으면 그대로 로드하고
	없으면 CIFAR-10 데이터셋(PIL->Tensor 변환)을 새로 만들어 저장합니다.
	"""


	# 만약 cache_path를 지정하지 않았다면, train 여부에 따라 기본 경로를 다르게 설정
	if cache_path is None:
		cache_path = "./cifar10_cache_train.pt" if train else "./cifar10_cache_test.pt"

	if os.path.exists(cache_path):
		images, targets = torch.load(cache_path)
		return images, targets

	# 캐시 파일이 없는 경우, 새로 로드 및 변환
	raw_data = torchvision.datasets.CIFAR10(
		root="./data",
		train=train,
		download=True
	)

	images_list = []
	targets_list = []
	for img, label in raw_data:
		img_tensor = torchvision.transforms.ToTensor()(img)  # 0~1 범위로 정규화
		images_list.append(img_tensor)
		targets_list.append(label)

	images = torch.stack(images_list)
	targets = torch.tensor(targets_list)

	if flatten:
		images = images.view(images.size(0), -1)

	# 변환된 결과를 캐시로 저장
	torch.save((images, targets), cache_path)

	return images, targets



def plot_samples(train_dataloader):
	# 처음 10개 이미지 시각화
	images, labels = next(iter(train_dataloader))

	plt.close()
	fig, axes = plt.subplots(2, 5, figsize=(12, 5))
	for i in range(10):
		ax = axes[i // 5, i % 5]
		# (채널, 높이, 너비) 순서를 (높이, 너비, 채널)로 바꿈
		ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
		ax.set_title(f"Label: {labels[i].item()}")
		ax.axis('off')  # 축 숨기기

	plt.tight_layout()
	return fig