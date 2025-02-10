import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_subplots_figure(image_paths, nrows=1, ncols=None):
	"""
	주어진 이미지 파일 경로들의 list를 받아서 nrows x ncols 형태의 subplot에 순서대로 이미지를 표시한 뒤,
	Figure 객체를 반환하는 함수입니다. 저장은 함수 밖에서 직접 진행할 수 있습니다.

	Args:
		image_paths (list): 이미지 파일 경로들의 list.
		nrows (int, optional): subplot의 행 개수. 기본값은 1입니다.
		ncols (int, optional): subplot의 열 개수. 기본값은 None이며, 이 경우 len(image_paths)에 맞춰 자동으로 설정됩니다.

	Returns:
		matplotlib.figure.Figure: 생성된 subplot이 담긴 Figure 객체.
	"""
	if ncols is None:
		ncols = len(image_paths)

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

	# axes가 1차원, 2차원 등 다양한 형태가 될 수 있으므로 일관된 방식으로 처리
	# (nrows=1 또는 ncols=1인 경우, axes가 numpy array가 아닌 matplotlib.axes.Axes 객체가 될 수 있음)
	# 이를 일관된 구조로 다루기 위해 np.array로 형변환
	if nrows == 1 and ncols == 1:
		axes = [axes]
	else:
		import numpy as np
		axes = np.array(axes).reshape(-1)

	for i, img_path in enumerate(image_paths):
		img = mpimg.imread(img_path)
		axes[i].imshow(img)
		axes[i].axis('off')  # 축 라벨 제거

	plt.tight_layout()
	return fig
