# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""

import torch
from torchmetrics.classification import (
	BinaryROC,
	BinaryAUROC,
	BinaryPrecisionRecallCurve,
	BinaryAveragePrecision
)

def calculate_roc(data, name="test"):
	"""
	이 함수는 이진분류를 위한 (logits, label) 데이터를 입력받아
	ROC 커브 (FPR, TPR)와 AUC를 계산하여 반환한다.

	Args:
		data (dict):
			{
				"logits": (batch_size, 2) 형태의 PyTorch Tensor,
				"y":	  (batch_size,)	 형태의 PyTorch Tensor
			}
		name (str, optional):
			결과 딕셔너리에 붙일 suffix이다. 기본값은 "test"이다.

	Returns:
		dict:
			{
				f"fpr_{name}":		torch.Tensor (FPR 값들),
				f"tpr_{name}":		torch.Tensor (TPR 값들),
				f"thresholds_roc_{name}": torch.Tensor (ROC 커브용 Threshold들),
				f"auc_roc_{name}":	float (ROC-AUC 값)
			}
	"""
	logits = data["logits"]  # (batch_size, 2)
	y = data["y"]			# (batch_size,)

	# 이진분류인지 확인
	assert logits.ndim == 2 and logits.shape[1] == 2, (
		"logits.shape은 (N, 2)가 되어야 한다."
	)

	# class=1 에 해당하는 예측 확률 계산
	probs = torch.softmax(logits, dim=1)[:, 1]

	# 1) ROC 커브 (FPR, TPR)
	roc_metric = BinaryROC(thresholds=None)
	roc_metric.update(probs, y)
	fpr, tpr, thresholds_roc = roc_metric.compute()

	# 2) ROC AUC
	auroc_metric = BinaryAUROC()
	auroc_metric.update(probs, y)
	roc_auc = auroc_metric.compute()

	return {
		f"fpr_{name}": fpr,
		f"tpr_{name}": tpr,
		f"thresholds_roc_{name}": thresholds_roc,
		f"auc_roc_{name}": float(roc_auc.item())
	}


def calculate_pr(data, name="test"):
	"""
	이 함수는 이진분류를 위한 (logits, label) 데이터를 입력받아
	Precision-Recall (PR) 커브와 PR-AUC를 계산하여 반환한다.

	Args:
		data (dict):
			{
				"logits": (batch_size, 2) 형태의 PyTorch Tensor,
				"y":	  (batch_size,)	 형태의 PyTorch Tensor
			}
		name (str, optional):
			결과 딕셔너리에 붙일 suffix이다. 기본값은 "test"이다.

	Returns:
		dict:
			{
				f"precision_{name}":  torch.Tensor (Precision 값들),
				f"recall_{name}":	 torch.Tensor (Recall 값들),
				f"thresholds_pr_{name}": torch.Tensor (PR 커브용 Threshold들),
				f"auc_pr_{name}":	 float (PR-AUC 값)
			}
	"""
	logits = data["logits"]
	y = data["y"]

	# 이진분류인지 확인
	assert logits.ndim == 2 and logits.shape[1] == 2, (
		"logits.shape은 (N, 2)가 되어야 한다."
	)

	# class=1 에 해당하는 예측 확률 계산
	probs = torch.softmax(logits, dim=1)[:, 1]

	# 1) Precision-Recall 커브
	pr_curve_metric = BinaryPrecisionRecallCurve()
	pr_curve_metric.update(probs, y)
	precision, recall, thresholds_pr = pr_curve_metric.compute()

	# 2) Average Precision (PR-AUC)
	ap_metric = BinaryAveragePrecision()
	ap_metric.update(probs, y)
	pr_auc = ap_metric.compute()

	return {
		f"precision_{name}": precision,
		f"recall_{name}": recall,
		f"thresholds_pr_{name}": thresholds_pr,
		f"auc_pr_{name}": float(pr_auc.item())
	}


def main():
	"""
	예시 main 함수: training과 testing을 반복 수행한 뒤,
	테스트 세트로부터 ROC와 PR 커브, 그리고 AUC를 계산한다.
	이 예시는 실제 동작보다는 함수 사용 방법을 간단히 보여주는 용도이다.
	"""
	# trainer, metric_tracker 등은 사용자 구현에 따라 정의되어 있다고 가정한다.
	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):
		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		from util_sac.pytorch.metrics.binary_roc_pr import calculate_roc, calculate_pr
		roc_test = calculate_roc(test_data, name="test")
		pr_test = calculate_pr(test_data, name="test")

		# metric_tracker에 기록
		mt.update(
			epoch,
			**train_loss,
			**test_loss,
			**roc_test,
			**pr_test
		)
		mt.print_latest()

	df = mt.generate_df()
	best_epoch_info = df.loc[df["auc_roc_test"].idxmax()].to_dict()
	print(best_epoch_info)


if __name__ == "__main__":
	main()
