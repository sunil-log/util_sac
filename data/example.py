import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from util_sac.data.trial_manager2 import trial_manager
import random
from sklearn.metrics import accuracy_score, f1_score

from util_sac.data.epoch_metric_tracker import metric_tracker
from util_sac.data.batch_metric_tracker import batch_loss_tracker

from torch.utils.data import TensorDataset, DataLoader


def generate_data(num_samples):
	data = []
	labels = []
	for _ in range(num_samples):
		sample = torch.zeros(10)
		indices = random.sample(range(10), 2)
		sample[indices] = 1
		label = torch.tensor([True if abs(indices[0] - indices[1]) == 1 else False], dtype=torch.float32)
		data.append(sample)
		labels.append(label)
	return torch.stack(data), torch.stack(labels)

class SimpleFCModel(nn.Module):
	def __init__(self):
		super(SimpleFCModel, self).__init__()
		self.fc1 = nn.Linear(10, 10)
		self.fc2 = nn.Linear(10, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.fc1(x)
		x = self.sigmoid(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return x


if __name__ == '__main__':
	sub_dir_list = ["weights", "reconstruction", "latent_space"]
	tm = trial_manager(sub_dir_list, trial_name="metric tracker test")

	# 데이터 생성
	train_data, train_labels = generate_data(2000)
	test_data, test_labels = generate_data(1000)
	"""
	train_data: torch.Size([3000, 10]), 
	train_labels: torch.Size([3000, 1])
	test_data: torch.Size([200, 10]), 
	test_labels: torch.Size([200, 1])
	"""

	# 데이터를 TensorDataset으로 변환
	train_dataset = TensorDataset(train_data, train_labels)
	test_dataset = TensorDataset(test_data, test_labels)

	# DataLoader 생성
	batch_size = 512  # 배치 크기를 지정합니다. 필요에 따라 조정하세요.
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



	# 모델 초기화
	model = SimpleFCModel()

	# 손실 함수와 옵티마이저 정의
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)


	# 학습 루프
	num_epochs = 500
	mt = metric_tracker()

	for epoch in range(num_epochs):
		model.train()
		loss_track = batch_loss_tracker()

		for batch_data, batch_labels in train_loader:
			# 순전파
			outputs = model(batch_data)
			loss = criterion(outputs, batch_labels)
			loss_track.update({"train_loss": loss})

			# 역전파 및 최적화
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		# 검증
		model.eval()
		with torch.no_grad():
			for batch_data, batch_labels in test_loader:
				outputs = model(batch_data)
				loss = criterion(outputs, batch_labels)
				loss_track.update({"val_loss": loss})

		mt.update(epoch=epoch, **loss_track.average())

		# print
		if epoch%50 == 0:
			mt.print_latest()

			plt.close()
			fig, ax = plt.subplots(1, 2, figsize=(15, 6))
			ax[0].plot(*mt["train_loss"], label="train_loss")
			ax[0].plot(*mt["val_loss"], label="val_loss")
			ax[0].legend()
			ax[0].set_title("Loss")
			ax[0].set_xlabel("Epoch")
			ax[0].set_ylabel("Loss")

			ax[1].plot(*mt["train_loss"], label="train_loss")
			ax[1].plot(*mt["val_loss"], label="val_loss")
			ax[1].legend()
			ax[1].set_title("Loss")
			ax[1].set_xlabel("Epoch")
			ax[1].set_ylabel("Loss")
			plt.tight_layout()
			plt.savefig(f"{tm.trial_dir}/loss_plot.png")












