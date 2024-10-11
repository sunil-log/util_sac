
# -*- coding: utf-8 -*-
"""
Created on  Oct 11 2024

@author: sac
"""


import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from util_sac.pandas.print_df import print_partial_markdown
from util_sac.data.print_array_info import print_array_info
from util_sac.image_processing.reduce_palette import reduce_palette_from_matplotlib_image

from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve



def pred_by_threshold(data, threshold):
	# calculate metrics; data["logit"].shape = (subject, 2)
	prob = softmax(data["logit"], axis=1)
	prob = prob[:, 1]
	pred = prob > threshold
	label = data["label"].tolist()

	return {"prob": prob, "pred": pred, "label": label}


def find_threshold_based_on_roc(data):
	prob = softmax(data["logit"], axis=1)
	prob = prob[:, 1]
	label = data["label"]

	# AUC 계산 가능 여부 확인
	try:
		auc = roc_auc_score(label, prob)

		# AUC가 0 또는 1인 경우 (완벽한 분류 또는 완벽한 오분류)
		if auc == 0 or auc == 1:
			return 0.5, 0  # 임계값 0.5, J 통계량 0 반환

		# ROC 곡선 계산
		fpr, tpr, thresholds = roc_curve(label, prob)

		# Youden's J statistic을 사용한 최적 임계값 찾기
		j_scores = tpr - fpr
		best_idx = np.argmax(j_scores)
		best_threshold = thresholds[best_idx]

		return best_threshold, j_scores[best_idx]

	except ValueError:
		# AUC 계산 불가능한 경우 (예: 모든 샘플이 같은 클래스로 예측된 경우)
		return 0.5, 0  # 임계값 0.5, J 통계량 0 반환


def calculate_accuracy(d):
	res = {}
	for k, v in d.items():
		accuracy = accuracy_score(v["label"], v["pred"])
		res[f"{k}_accuracy"] = accuracy
	return res

def calculate_f1(d, mode="binary"):
	"""
	mode = "binary", "micro", "macro", "weighted"
	"""
	res = {}
	for k, v in d.items():
		f1 = f1_score(v["label"], v["pred"], average=mode)
		res[f"{k}_{mode}_f1"] = f1
	return res


def calculate_auc(d):
	res = {}
	for k, v in d.items():
		try:
			# 레이블이 모두 같은 경우 처리
			if len(np.unique(v["label"])) == 1:
				res[f"{k}_auc"] = 0
			else:
				auc = roc_auc_score(v["label"], v["prob"])
				res[f"{k}_auc"] = auc
		except ValueError:
			# AUC 계산 불가능한 경우
			res[f"{k}_auc"] = 0
	return res

def calculate_confusion_matrix(d):
	res = {}
	for k, v in d.items():
		cm = confusion_matrix(v["label"], v["pred"])
		res[f"{k}_confusion_matrix"] = cm
	return res


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



def binary_metrics(train_data, valid_data, test_data):

	"""
	train_data, valid_data, test_data 를 받는다.
	각각 dict 로 "logit" 과 "label" 이 포함되어 있다.

	Valid set 을 이용해서 threshold 를 구한다.
	train, valid, test 의 prob,
	accuracy, f1, auc 를 구한다.
	confusion matrix 를 구한다.
	"""


	# find the best threshold using validation data
	th, j = find_threshold_based_on_roc(valid_data)

	# prob, pred, label 을 구한다; 각각은 "prob", "pred", "label" 을 포함한다.
	d_train = pred_by_threshold(train_data, th)
	d_valid = pred_by_threshold(valid_data, th)
	d_test = pred_by_threshold(test_data, th)
	d = {"train": d_train, "valid": d_valid, "test": d_test}

	# accuracy
	res_acc = calculate_accuracy(d)
	res_f1 = calculate_f1(d, mode="binary")
	res_auc = calculate_auc(d)
	res_conf_mat = calculate_confusion_matrix(d)
	"""
	{'train_accuracy': 0.3, 'valid_accuracy': 0.4, 'test_accuracy': 0.4}
	{'train_binary_f1': 0.0, 'valid_binary_f1': 0.0, 'test_binary_f1': 0.0}
	{'train_auc': 0.714, 'valid_auc': 0.166, 'test_auc': 0.354}
	{'train_confusion_matrix': array([[ 6,  0], [14,  0]] ...}
	"""

	return {**res_acc, **res_f1, **res_auc, "threshold": th}, res_conf_mat



def main():

	train_data = {
		"logit": np.random.randn(100, 2),
		"label": np.random.randint(0, 2, size=(100,))
	}

	valid_data = {
		"logit": np.random.randn(100, 2),
		"label": np.random.randint(0, 2, size=(100,))
	}

	test_data = {
		"logit": np.random.randn(100, 2),
		"label": np.random.randint(0, 2, size=(100,))
	}

	res_mat, conf_mat = binary_metrics(train_data, valid_data, test_data)
	plot_confusion_matrix(conf_mat["valid_confusion_matrix"], ax[0], f1=0.82)
	plot_confusion_matrix(conf_mat["test_confusion_matrix"], ax[1], f1=0.78)


if __name__ == "__main__":
	main()
