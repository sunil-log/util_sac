import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def create_subplots_figure(image_paths, ncols, nrows=None, figsize=None):
	"""
	주어진 이미지 파일 경로들의 list를 받아서 nrows x ncols 형태의 subplot에 순서대로 이미지를 표시한 뒤,
	Figure 객체를 반환하는 함수입니다. 저장은 함수 밖에서 직접 진행할 수 있습니다.

	Args:
		image_paths (list): 이미지 파일 경로들의 list.
		ncols (int): subplot의 열 개수(필수 인자).
		nrows (int, optional): subplot의 행 개수. 기본값은 None이며, 이 경우 len(image_paths)와 ncols에 맞춰 자동으로 설정됩니다.
		figsize (tuple, optional): 전체 Figure의 크기를 지정하는 (width, height) 형태의 튜플.
								   기본값은 (4 * ncols, 4 * nrows)입니다.

	Returns:
		matplotlib.figure.Figure: 생성된 subplot이 담긴 Figure 객체.
	"""
	# nrows가 지정되지 않았으면 이미지 개수와 ncols를 바탕으로 자동 계산
	if nrows is None:
		nrows = math.ceil(len(image_paths) / ncols)

	# figsize 기본값 설정
	if figsize is None:
		figsize = (4 * ncols, 4 * nrows)

	# subplot 생성
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

	# axes가 1차원, 2차원 등 다양한 형태가 될 수 있으므로 일관된 방식으로 처리
	# (nrows=1 또는 ncols=1인 경우, axes가 numpy array가 아닌 단일 matplotlib.axes.Axes 객체가 될 수 있음)
	if nrows == 1 and ncols == 1:
		axes = [axes]  # 단일 객체를 리스트로 변환
	else:
		axes = np.array(axes).reshape(-1)  # 다차원을 1차원으로 펴서 일관되게 인덱싱

	# 이미지 경로에 따라 순서대로 표시
	for i, img_path in enumerate(image_paths):
		img = mpimg.imread(img_path)
		axes[i].imshow(img)
		axes[i].axis('off')  # 축 라벨 제거

	plt.tight_layout()
	return fig


if __name__ == '__main__':

	# search files
	root_directory = "./trials/"
	search_pattern = "__onlyPD"
	df = search_items_df(root_directory, search_pattern, search_type='directories')
	"""
	|    | Path                                                             | Parent   | Name                                                      | Type      | Extension   |
	|---:|:-----------------------------------------------------------------|:---------|:----------------------------------------------------------|:----------|:------------|
	|  0 | trials/2025-02-10_07-03-21__REM__epoch_30s__6th_location__onlyPD | trials   | 2025-02-10_07-03-21__REM__epoch_30s__6th_location__onlyPD | Directory |             |
	|  1 | trials/2025-02-10_07-02-51__N3__epoch_30s__6th_location__onlyPD  | trials   | 2025-02-10_07-02-51__N3__epoch_30s__6th_location__onlyPD  | Directory |             |
	|  2 | trials/2025-02-10_07-01-15__N1__epoch_30s__6th_location__onlyPD  | trials   | 2025-02-10_07-01-15__N1__epoch_30s__6th_location__onlyPD  | Directory |             |
	|  3 | trials/2025-02-10_07-01-55__N2__epoch_30s__6th_location__onlyPD  | trials   | 2025-02-10_07-01-55__N2__epoch_30s__6th_location__onlyPD  | Directory |             |
	"""

	# sort by Name
	df = df.sort_values(by='Name')

	# image path
	df['image'] = df['Path'].apply(lambda x: list(x.glob("*umap__*.png"))[0])

	# create subplots figure
	fig = create_subplots_figure(df['image'], ncols=2, figsize=(16, 16))
	img = reduce_palette(fig, 32)
	img.save("output.png")



