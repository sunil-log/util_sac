
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""


from torchmetrics import F1Score


def calculate_f1(data, name="test"):

	# Extract Data
	logits = data["logits"]
	y = data["y"]
	"""
	logits     PyTorch Tensor       (1442, 2)                   11.27 KB torch.float32
	y          PyTorch Tensor       (1442,)                     11.27 KB torch.int64
	"""

	# 예측 라벨 계산 (가장 높은 값의 인덱스를 예측으로 사용)
	y_pred = logits.argmax(dim=1)

	# 전체 클래스 수 (logits의 두 번째 차원과 동일하다고 가정)
	num_classes = logits.shape[1]

	# 1) 각 클래스별 F1Score 계산 (average=None)
	f1_metric_per_class = F1Score(
	    task="multiclass",
	    average=None,
	    num_classes=num_classes
	)
	f1_per_class = f1_metric_per_class(y_pred, y)
	"""
	tensor([0.8598, 0.0979])
	"""
	print(f1_per_class)

	# 2) 전체 클래스에 대해 Macro average F1Score 계산
	f1_metric_macro = F1Score(
		task="multiclass",
		average="macro",
		num_classes=num_classes
	)
	f1_macro = f1_metric_macro(y_pred, y)
	"""
	tensor(0.4789)
	"""

	# make d
	d = {}
	for i in range(num_classes):
		d[f"f1_class_{i}_{name}"] = f1_per_class[i].item()
	d[f"f1_class_macro_{name}"] = f1_macro.item()

	return d




def main():


	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):
		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		print_array_info(train_data)
		"""
		logits     PyTorch Tensor       (1442, 2)                   11.27 KB torch.float32
		y          PyTorch Tensor       (1442,)                     11.27 KB torch.int64
		"""

		from util_sac.metrics.multi_class_matrics import calculate_f1
		f1_train = calculate_f1(train_data, name="train")
		f1_test = calculate_f1(test_data, name="test")

		mt.update(epoch, **train_loss, **test_loss, **f1_train, **f1_test)
		mt.print_latest()




if __name__ == "__main__":
	main()
