
# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2025

@author: sac
"""


def plot_confusion_matrix(cm, ax, f1=0.5):
	"""
	Draws a confusion matrix on the given Axes object.

	Parameters:
	cm (numpy.ndarray): Confusion matrix.
	ax (matplotlib.axes.Axes): Axes object to draw the confusion matrix on.
	"""
	cax = ax.imshow(cm, cmap='Blues', interpolation='none')
	ax.figure.colorbar(cax, ax=ax)
	ax.set_title(f'Confusion Matrix, Binary F1: {f1:.3f}')

	# 축 레이블 설정
	tick_marks = np.arange(len(['True', 'False']))
	ax.set_xticks(tick_marks)
	ax.set_xticklabels(['Predicted False', 'Predicted True'])
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(['True False', 'True True'])

	# 텍스트 주석 추가
	thresh = cm.mean()
	for i, j in np.ndindex(cm.shape):
		color = "white" if cm[i, j] > thresh else "black"
		ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color)

	ax.set_ylabel('True label')
	ax.set_xlabel('Predicted label')
	ax.figure.tight_layout()
