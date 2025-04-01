
# -*- coding: utf-8 -*-
"""
Created on  Oct 10 2024

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette_from_matplotlib_image


def pad_and_mask(data, target_length):
	"""
	pad_and_mask() 함수는 Numpy array 형태의 data에 대해, 0번째 dimension(길이)을 기준으로
	zero-padding 또는 잘라냄을 수행한다. 이후 원본 data가 존재하는 부분을 1, padding된 부분을 0으로
	표시하는 mask를 함께 반환한다.

	Dimension 구조
	--------------
	data는 임의의 차원 구조를 가질 수 있다. 단, 0번째 dimension(길이)에 대해서만 padding 또는 잘라냄을 수행한다.
	예를 들어, data가 (sequence_length, channel), (sequence_length, channel, height, width),
	(sequence_length, channel, height, width, depth) 등 어떤 형태든 상관없이 적용 가능하다.

	Parameters
	----------
	data : np.ndarray
	    0번째 dimension의 길이가 변동될 수 있는 Numpy array이다.
	    예: (sequence_length, channel, height, width) 등 자유로운 형태.
	target_length : int
	    맞추고자 하는 목표 길이이다. data의 0번째 dimension을 기준으로 zero-padding하거나 잘라낸다.

	Returns
	-------
	padded_data : np.ndarray
	    target_length에 맞추어 zero-padding 또는 잘라낸 결과물이다.
	    shape은 (target_length, ...) 형태가 된다.
	mask : np.ndarray
	    원본 data가 존재하는 부분은 1, padding된 부분은 0으로 표시되는 mask이다.
	    shape은 padded_data와 동일하다.

	Notes
	-----
	- 원본 data의 길이가 target_length보다 길면 초과분을 잘라낸다.
	- 원본 data의 길이가 target_length보다 짧으면 부족분만큼 zero-padding한다.
	- mask는 원본 data가 실제로 존재하는 위치를 1로 표시하며, padding된 부분은 0으로 표시한다.
	  모델 입력 시 유효한 시퀀스 길이를 구분하거나, 후처리에 활용할 수 있다.
	"""

	# determine pad width
	original_shape = data.shape
	pad_0dim =(0, max(0, target_length - original_shape[0]))  # (0, len_pad)
	pad_1dim = [(0, 0)] * (len(original_shape) - 1) # (n_dim - 1) * (0, 0)
	pad_width = [pad_0dim] + pad_1dim
	"""
	[(0, 130), (0, 0), (0, 0), (0, 0)]
		dim 0 에서 앞에는 0개 뒤에는 130개를 패딩한다.
		나머지 차원은 패딩하지 않는다.
	"""

	padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
	mask = np.pad(np.ones_like(data), pad_width, mode='constant', constant_values=0)

	if original_shape[0] > target_length:
		padded_data = padded_data[:target_length]
		mask = mask[:target_length]

	return padded_data, mask



def plot_pad_and_mask(data, mask, n_batch):

	fig, ax = plt.subplots(2, 1, figsize=(13, 5))

	# Plot EMG data
	im1 = ax[0].imshow(data[:, 0, 0, :].T, aspect='auto', cmap='Blues')
	ax[0].set_title('Data')
	fig.colorbar(im1, ax=ax[0], label='Intensity')

	# Plot mask data
	im2 = ax[1].imshow(mask[:, 0, 0, :].T, aspect='auto', cmap='Blues')
	ax[1].set_title('Mask Data')
	fig.colorbar(im2, ax=ax[1], label='Mask Value')

	# Add vertical line at the end of n_batch
	ax[0].axvline(x=n_batch - 1, color='r', linestyle='--', linewidth=1)
	ax[1].axvline(x=n_batch - 1, color='r', linestyle='--', linewidth=1)

	# Adjust layout and save figure
	plt.tight_layout()
	return fig




if __name__ == "__main__":
	pass