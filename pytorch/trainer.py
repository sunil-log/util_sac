import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Optional
import numpy as np
from abc import ABC, abstractmethod


class MLTrainer(ABC):
	def __init__(
			self,
			model: nn.Module,
			train_loader: DataLoader,
			test_loader: DataLoader,
			optimizer: torch.optim.Optimizer,
			criterion: Callable,
			device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
	):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.model.to(self.device)

	@abstractmethod
	def train_step(self, batch):
		pass

	@abstractmethod
	def test_step(self, batch):
		pass

	def train_epoch(self):
		self.model.train()
		total_loss = 0
		for batch in self.train_loader:
			self.optimizer.zero_grad()
			loss = self.train_step(batch)
			loss.backward()
			self.optimizer.step()
			total_loss += loss.item()
		return total_loss / len(self.train_loader.dataset)

	def test_epoch(self):
		self.model.eval()
		total_loss = 0
		with torch.no_grad():
			for batch in self.test_loader:
				loss = self.test_step(batch)
				total_loss += loss.item()
		return total_loss / len(self.test_loader.dataset)

	def train(self, num_epochs: int, save_path: Optional[str] = None):
		for epoch in range(num_epochs):
			train_loss = self.train_epoch()
			test_loss = self.test_epoch()
			print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

			if save_path:
				torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch + 1}.pth")

		return train_loss, test_loss