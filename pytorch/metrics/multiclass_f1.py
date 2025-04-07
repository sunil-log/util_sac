
# -*- coding: utf-8 -*-
"""
Created on  Feb 23 2025

@author: sac
"""


from torchmetrics import F1Score


def calculate_f1s(train_data, valid_data, test_data):
	f1_train = calculate_f1(train_data, name="train")
	f1_valid = calculate_f1(valid_data, name="valid")
	f1_test = calculate_f1(test_data, name="test")

	# concat all f1s
	f1s = {**f1_train, **f1_valid, **f1_test}
	return f1s


def calculate_f1(data, name="test"):
	"""
	이 함수는 입력으로 주어진 data에서 logits와 레이블 y를 추출한 뒤,
	예측 라벨을 계산하고 PyTorch의 torchmetrics 라이브러리를 활용하여
	Multiclass F1Score를 산출한 뒤 딕셔너리 형태로 반환한다.

	Args:
	    data (dict):
	    {
			"logits": (batch_size, num_classes) 형태의 PyTorch Tensor,
			"y":      (batch_size,)             형태의 PyTorch Tensor
		}
	    name (str, optional):
	        결과 딕셔너리에 부가적으로 붙일 suffix이다. 기본값은 "test"이다.

	Returns:
	    dict:
	        {
	            "f1_class_0_{name}": float 값,
	            "f1_class_1_{name}": float 값,
	            ...,
	            "f1_class_macro_{name}": float 값
	        }
	        과 같은 형태로 각 클래스별 F1Score와 macro average F1Score를 모두 담는다.
	"""

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
	"""
	이 메인 함수는 training과 testing을 반복 수행한 뒤,
	train_data와 test_data를 통해 F1Score를 계산한다.

	train_data, test_data는 다음과 같은 key와 구조를 가진 dict여야 한다:
		{
			"logits": (batch_size, num_classes) 형태의 PyTorch Tensor,
			"y":      (batch_size,)             형태의 PyTorch Tensor
		}
	"""

	mt = metric_tracker()
	for epoch in range(trainer.n_epoch):
		train_loss, train_data = trainer.one_epoch(mode='train', epoch=epoch)
		test_loss, test_data = trainer.one_epoch(mode='test', epoch=epoch)

		print_array_info(train_data)
		"""
		logits     PyTorch Tensor       (1442, 2)                   11.27 KB torch.float32
		y          PyTorch Tensor       (1442,)                     11.27 KB torch.int64
		"""

		from util_sac.pytorch.metrics.multiclass_f1 import calculate_f1
		f1_train = calculate_f1(train_data, name="train")
		f1_test = calculate_f1(test_data, name="test")

		mt.update(epoch, **train_loss, **test_loss, **f1_train, **f1_test)
		mt.print_latest()


	df = mt.generate_df()
	max_f1_dict = df.loc[df["f1_class_macro_test"].idxmax()].to_dict()
	print(max_f1_dict)

if __name__ == "__main__":
	main()
