import torch
import torch.nn as nn


class TwoLayerMNIST(nn.Module):
	def __init__(self, args):
		super().__init__()
		arm = args["model"]  # args["model"]를 arm으로 할당
		self.fc1 = nn.Linear(arm["input_dim"], arm["hidden_dim"])
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(arm["dropout"])
		self.fc2 = nn.Linear(arm["hidden_dim"], arm["output_dim"])

	def forward(self, x):
		# MNIST 이미지는 28×28이므로, (batch_size, 784)로 Flatten한다
		x = x.view(-1, 784)

		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
