
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

def show_padding(z_pad, z_mask):
	"""
	z_pad: PyTorch Tensor (1000, 12, 64), torch.float32
	z_mask: PyTorch Tensor (1000,), torch.int32

	이 함수는 z_pad와 z_mask를 각각 imshow로 시각화하는
	위아래로 배치된 2개의 Subplot으로 구성된 Figure를 생성하고 반환한다.
	각 Subplot에는 colorbar를 추가하고, color 범위는 data의 mean ± std(1 sigma)로 설정한다.
	"""

	# PyTorch Tensor를 NumPy array로 변환
	z_pad_np = z_pad.detach().cpu().numpy()
	z_pad_np = z_pad_np[:, 0, :].T  # (64, 1000)

	z_mask_np = z_mask.detach().cpu().numpy()
	z_mask_np = z_mask_np.reshape(-1, 1).T  # (1, 1000)

	# z_pad_np의 mean과 std 계산
	z_pad_mean = np.mean(z_pad_np)
	z_pad_std = np.std(z_pad_np)
	vmin_pad = z_pad_mean - z_pad_std
	vmax_pad = z_pad_mean + z_pad_std

	# z_mask_np의 mean과 std 계산
	z_mask_mean = np.mean(z_mask_np)
	z_mask_std = np.std(z_mask_np)
	vmin_mask = z_mask_mean - z_mask_std
	vmax_mask = z_mask_mean + z_mask_std

	# Figure와 Subplot(위아래 배치) 생성
	plt.close()
	fig, axes = plt.subplots(2, 1, figsize=(16, 8))

	# 첫 번째 Subplot - z_pad
	im0 = axes[0].imshow(z_pad_np, aspect='auto', vmin=vmin_pad, vmax=vmax_pad, interpolation='none')
	axes[0].set_title("z_pad")
	fig.colorbar(im0, ax=axes[0])

	# 두 번째 Subplot - z_mask
	im1 = axes[1].imshow(z_mask_np, aspect='auto', vmin=vmin_mask, vmax=vmax_mask, interpolation='none')
	axes[1].set_title("z_mask")
	fig.colorbar(im1, ax=axes[1])

	plt.tight_layout()
	return fig