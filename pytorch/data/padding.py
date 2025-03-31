
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
from util_sac.pytorch.data.print_array_info import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette_from_matplotlib_image


def pad_and_mask(data, target_length):

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



def main():
	pass

if __name__ == "__main__":
	main()
