from PIL import Image
import numpy as np
import io

def reduce_palette(figure, palette_size):
	"""
	Matplotlib으로 그려진 이미지의 컬러 팔레트를 지정된 크기로 줄이는 함수.

	Args:
		image (matplotlib.figure.Figure): Matplotlib으로 그려진 이미지.
		palette_size (int): 지정된 크기로 줄일 팔레트의 색상 수.

	Returns:
		(PIL.Image.Image): 처리된 이미지.

	Example:
		fig = plt.figure(figsize=(5, 5))
		ax = fig.add_subplot(111)

		# 이미지 그리기
		# ...

		result_image = reduce_palette_from_matplotlib_image(fig, 16)
		result_image.save('result.png')
	"""
	# Figure 객체에서 PNG 데이터로 변환
	buf = io.BytesIO()
	figure.savefig(buf, format='png', bbox_inches='tight')
	buf.seek(0)

	# PNG 데이터를 NumPy 배열로 변환
	data = np.array(Image.open(buf))

	# 컬러 팔레트 줄이기
	pil_image = Image.fromarray(data)
	pil_image = pil_image.quantize(colors=palette_size)

	return pil_image