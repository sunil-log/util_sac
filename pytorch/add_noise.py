import torch

def add_normal_noise(tensor, std):
	"""
	주어진 텐서에 정규 분포 노이즈를 추가하는 함수

	Args:
		tensor (torch.Tensor): 노이즈를 추가할 텐서
		std (float): 정규 분포의 표준편차

	Returns:
		torch.Tensor: 노이즈가 추가된 텐서
	"""
	noise = torch.randn_like(tensor) * std
	noisy_tensor = tensor + noise
	return noisy_tensor