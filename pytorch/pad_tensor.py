
import torch
import matplotlib.pyplot as plt


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
	주어진 z_pad와 z_mask를 받아서 padding 결과를 시각화합니다.

	z_pad는 (pad_size, a, b, c, ...) 형태일 수 있으므로, 배치 차원 이후의 모든 차원을 flatten하여
	(pad_size, -1) 형태의 2차원 배열로 변환한 후 imshow로 그립니다.

	z_mask는 (pad_size,) 형태이므로, (1, pad_size)로 reshape한 후 imshow로 그려서,
	원래 데이터 영역(1)과 padding 영역(0)이 올바르게 구성되었는지 확인할 수 있습니다.

	Parameters:
		z_pad (torch.Tensor): padding이 적용된 텐서, shape = (pad_size, a, b, c, ...)
		z_mask (torch.Tensor): mask 텐서, shape = (pad_size,), 원래 데이터 영역은 1, padding 영역은 0
	"""
	# 텐서를 numpy 배열로 변환 (배치 차원 이후의 모든 차원을 flatten)
	z_pad_vis = z_pad.view(z_pad.size(0), -1).detach().cpu().numpy()
	z_mask_vis = z_mask.view(1, -1).detach().cpu().numpy()

	# 시각화를 위한 서브플롯 생성
	plt.close()
	fig, axes = plt.subplots(2, 1, figsize=(12, 6))

	# z_pad 시각화
	im1 = axes[0].imshow(z_pad_vis, aspect='auto', interpolation='nearest')
	axes[0].set_title('z_pad visualization')
	plt.colorbar(im1, ax=axes[0])

	# z_mask 시각화 (흑백 cmap 사용)
	im2 = axes[1].imshow(z_mask_vis, aspect='auto', interpolation='nearest', cmap='gray')
	axes[1].set_title('z_mask visualization')
	plt.colorbar(im2, ax=axes[1])

	plt.tight_layout()
	return fig

