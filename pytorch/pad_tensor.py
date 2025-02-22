
import torch
import numpy as np
import matplotlib.pyplot as plt
from util_sac.data.print_array_info import print_array_info


def pad_tensor(z, pad_size, pad_value=0):
	"""
	주어진 텐서 z (batch, a, b, c, d, ...)에 대해, pad_size를 맞추는 함수를 만듭니다.
	만약 pad_size가 원래 batch 크기보다 작으면, 단순히 앞부분을 잘라냅니다.
	pad_size가 원래 batch 크기보다 크면, 부족한 부분을 pad_value로 채웁니다.
	이에 따라, z_mask 텐서는 원래 데이터 부분은 1, padding 영역은 0으로 구성됩니다.

	Parameters:
		z (torch.Tensor): 원래 텐서, shape = (batch, a, b, c, d, ...)
		pad_size (int): 결과 텐서의 batch 크기 (예: 1000)
		pad_value (scalar, optional): padding할 때 사용할 값, default는 0

	Returns:
		z_pad (torch.Tensor): 결과 텐서, shape = (pad_size, a, b, c, d, ...)
		z_mask (torch.Tensor): mask 텐서, shape = (pad_size,), 원래 데이터 부분은 1, padding 부분은 0
	"""
	batch_size = z.size(0)
	if pad_size < batch_size:
		# pad_size가 작으면 앞부분만 잘라냅니다.
		z_pad = z[:pad_size]
		z_mask = torch.ones(pad_size, dtype=torch.int, device=z.device)
	elif pad_size == batch_size:
		z_pad = z.clone()
		z_mask = torch.ones(batch_size, dtype=torch.int, device=z.device)
	else:
		# pad_size가 크면 padding 영역을 pad_value로 채워 넣습니다.
		pad_shape = (pad_size,) + z.shape[1:]
		z_pad = torch.full(pad_shape, pad_value, dtype=z.dtype, device=z.device)
		z_pad[:batch_size] = z
		z_mask = torch.cat([
			torch.ones(batch_size, dtype=torch.int, device=z.device),
			torch.zeros(pad_size - batch_size, dtype=torch.int, device=z.device)
		], dim=0)

	return z_pad, z_mask


def show_multi_tensors(**kwargs):
	"""
	여러 개의 PyTorch Tensor를 **kwargs 형태로 받아,
	각각의 데이터를 Subplot으로 시각화하여 하나의 Figure로 반환합니다.

	용례:
		fig = show_multi_tensors(eeg=z_eeg, eog=z_eog, emg=z_emg, pad_mask=pad_mask)

	각 Tensor는 다음 형태 중 하나로 가정합니다:
	(1000,) 혹은 (1000, c), (1000, c, d) 등

	- 1차원 (1000,)일 경우: (1, 1000) 형태로 변환하여 시각화합니다.
	- 2차원 (1000, c)일 경우: (c, 1000) 형태로 변환(전치)하여 시각화합니다.
	- 3차원 (1000, c, d)일 경우: 두 번째 축(채널 중 첫 번째)만 사용 후 전치하여 (d, 1000) 형태로 시각화합니다.

	시각화 범위(vmin, vmax)는 각 데이터의 mean ± std(1 sigma)로 설정하고,
	데이터마다 colorbar를 추가합니다.
	"""
	plt.close()
	fig, axes = plt.subplots(len(kwargs), 1, figsize=(16, 3 * len(kwargs)), squeeze=False)

	for i, (key, val) in enumerate(kwargs.items()):
		data = val.detach().cpu().numpy()

		# 데이터 형태에 따라 2D 형태로 변환
		if data.ndim == 1:  # (1000,)
			data = data.reshape(1, -1)  # (1, 1000)
		elif data.ndim == 2:  # (1000, c)
			data = data.T  # (c, 1000)
		elif data.ndim == 3:  # (1000, c, d)
			data = data[:, 0, :].T  # (d, 1000)

		# mean ± std 범위 설정
		d_mean = np.mean(data)
		d_std = np.std(data)
		vmin = d_mean - d_std
		vmax = d_mean + d_std

		im = axes[i, 0].imshow(data, aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')
		axes[i, 0].set_title(key)
		fig.colorbar(im, ax=axes[i, 0])

	plt.tight_layout()
	return fig