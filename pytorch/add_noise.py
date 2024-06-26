import torch


def add_normal_noise_to_tensor(tensor, noise_std):
	"""
	주어진 텐서에 정규 분포 노이즈를 추가하는 함수 (인플레이스 연산)

	Args:
		tensor (torch.Tensor): 노이즈를 추가할 텐서
		std (float): 정규 분포의 표준편차

	Examples:
		eeg = add_normal_noise(eeg, std=0.1)
		eog = add_normal_noise(eog, std=0.05)
	"""
	noise = torch.randn_like(tensor) * noise_std
	tensor.add_(noise)



def inject_noise_to_parameters(module, noise_std):
	"""
	모듈 내부의 모든 Parameter에 Gaussian 노이즈를 주입하는 함수

	Args:
		module (torch.nn.Module): 노이즈를 주입할 모듈
		noise_std (float): 주입할 Gaussian 노이즈의 표준편차

	Example:
		inject_noise_to_parameters(model.attention_1, noise_std=0.01)
	"""
	with torch.no_grad():  # 노이즈 주입 시 gradient 계산 방지
		for param in module.parameters():
			noise = torch.randn_like(param) * noise_std
			param.add_(noise)  # in-place 연산으로 노이즈 추가

