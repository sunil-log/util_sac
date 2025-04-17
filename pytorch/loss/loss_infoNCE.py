
# -*- coding: utf-8 -*-
"""
Created on  Jan 29 2025

@author: sac
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
	def __init__(self, temperature=0.5):
		super(SupervisedContrastiveLoss, self).__init__()
		self.temperature = temperature

	def forward(self, features, labels):
		"""
		Computes the supervised contrastive loss.

		Args:
			features: Tensor of shape [batch_size, feature_dim]
			labels: Tensor of shape [batch_size]
		Returns:
			loss: Scalar tensor representing the loss
		"""
		device = features.device

		"""
		2. pair label and mask
		"""
		labels = labels.contiguous().view(-1, 1)
		mask = torch.eq(labels, labels.T).float().to(device)
		logits_mask = (torch.ones_like(mask) - torch.eye(features.shape[0])).to(device)
		final_mask = mask * logits_mask

		"""
		3. cosine similarity
		"""
		features = F.normalize(features, p=2, dim=1)
		s = torch.matmul(features, features.T)

		"""
		4. Probability for Pair-Label
		"""
		s = s / self.temperature
		exp_s = torch.exp(s) * logits_mask
		log_p = s - torch.log(exp_s.sum(axis=1, keepdim=True) + 1e-12)

		"""
		5. Cross Entropy for Pair Label
		"""
		Z = final_mask.sum(1, keepdim=True).clamp(min=1e-8)
		CEs = -(final_mask / Z * log_p).sum(1)
		loss = CEs.mean()

		return loss


def main():

	# Example trials
	features = torch.tensor([
		[1.0, 2.0, 3.0, 4.0],
		[2.0, 3.0, 4.0, 5.0],
		[5.0, 4.0, 3.0, 2.0]
	], requires_grad=True)

	labels = torch.tensor([0, 0, 1])

	# Instantiate the loss module
	criterion = SupervisedContrastiveLoss(temperature=0.5)

	# Compute the loss
	loss = criterion(features, labels)

	print(f"Supervised Contrastive Loss: {loss.item()}")



if __name__ == '__main__':
	main()