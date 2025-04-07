import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch

from util_sac.pytorch.data.print_array import print_array_info
from util_sac.pytorch.dataloader.to_tensor_device import dict_to_tensors



def label_distribution_table(data, label_col="y"):
	import torch
	import pandas as pd

	splits = ["train", "valid", "test"]
	df_result = pd.DataFrame()

	for split in splits:
		if split not in data:
			continue

		# 데이터의 레이블을 PyTorch Tensor에서 NumPy 배열로 변환
		y_tensor = data[split][label_col]
		y_numpy = y_tensor.cpu().numpy()

		# split 하나에 대한 DataFrame 생성
		df_split = pd.DataFrame({label_col: y_numpy})

		# groupby로 레이블별 count 계산
		counts = df_split.groupby(label_col).size().rename(f"count_{split}")
		percents = (counts / counts.sum() * 100).rename(f"percent_{split}")

		# df_result에 순차적으로 병합
		if df_result.empty:
			# 아직 아무것도 없다면 counts, percents 함께 첫 테이블 구성
			df_result = pd.concat([counts, percents], axis=1)
		else:
			# 기존 테이블에 join으로 새 column들을 추가
			df_result = df_result.join(counts, how="outer")
			df_result = df_result.join(percents, how="outer")

	# 인덱스(레이블 값)를 'Class'로 변환
	df_result.index.name = "Class"
	df_result.reset_index(inplace=True)

	# NaN을 0으로 대체
	df_result.fillna(0, inplace=True)

	# 총합(Total) 행 만들기
	# (count_ 계열은 합, percent_ 계열은 합하면 대체로 100 근방이지만 그대로 합으로 보여준다)
	total_row = {"Class": "Total"}
	for col in df_result.columns:
		if col.startswith("count_") or col.startswith("percent_"):
			total_row[col] = df_result[col].sum()
		else:
			# Class 열은 "Total"이므로 무시
			continue

	# total_row를 DataFrame에 추가
	df_result = pd.concat([df_result, pd.DataFrame([total_row])], ignore_index=True)

	return df_result




def split_data_into_train_valid_test(
		data: dict,
		fold_i: int,
		fold_count: int,
		fold_seed: int = 42,
		stratify_key: str = None
):
	"""
	generate_train_test_index 함수를 사용하여 data를 train, valid, test로 분할한다.

	매개변수:
		data: key마다 (N, ...) 형태를 갖는 Python Dictionary
		fold_i: 현재 test로 사용할 fold index
		fold_count: 전체 fold 개수
		fold_seed: generate_fold_index 함수에서 사용할 random seed
		stratify_key: 특정 key(예: 'class') 기준으로 계층화 분할을 수행할 때 사용

	반환:
		data_train, data_valid, data_test: 각 split에 해당하는 Dictionary
	"""

	# move data to tensors
	data = dict_to_tensors(data)
	"""
	subject    PyTorch Tensor       (95,)                       760.00 B torch.int64
	residence  PyTorch Tensor       (95, 3)                      2.23 KB torch.int64
	gender     PyTorch Tensor       (95, 2)                      1.48 KB torch.int64
	age        PyTorch Tensor       (95,)                       380.00 B torch.float32
	RealVote   PyTorch Tensor       (95,)                       760.00 B torch.int64
	rt         PyTorch Tensor       (95, 5, 40)                 74.22 KB torch.float32
	listAns    PyTorch Tensor       (95, 5, 40)                148.44 KB torch.int64
	pressedKey PyTorch Tensor       (95, 5, 40)                148.44 KB torch.int64
	error      PyTorch Tensor       (95, 5, 40)                148.44 KB torch.int64
	StimRaw    PyTorch Tensor       (95, 5, 40)                148.44 KB torch.int64
	mask_valid PyTorch Tensor       (95, 5, 40)                148.44 KB torch.int64
	y_bin      PyTorch Tensor       (95,)                       760.00 B torch.int64
	word_emb   PyTorch Tensor       (95, 5, 40, 2)             296.88 KB torch.int64
	"""


	# generate_train_test_index를 통해 index를 구한다
	idx_dict = generate_train_test_index(
		data=data,
		fold_i=fold_i,
		fold_count=fold_count,
		fold_seed=fold_seed,
		stratify_key=stratify_key
	)

	train_idx = idx_dict["train"]
	valid_idx = idx_dict["valid"]
	test_idx = idx_dict["test"]

	data_train = {}
	data_valid = {}
	data_test = {}

	# data Dictionary의 각 key별로 분할을 수행한다
	for key, value in data.items():
		data_train[key] = value[train_idx]
		data_valid[key] = value[valid_idx]
		data_test[key] = value[test_idx]

	return {"train": data_train, "valid": data_valid, "test": data_test}


def generate_train_test_index(
	data: dict,
	fold_i: int,
	fold_count: int,
	fold_seed: int = 42,
	stratify_key: str = None
):
	"""
	generate_fold_index 함수를 이용하여 fold_i를 test로, (fold_i+1)%fold_count를
	valid로, 나머지를 전부 train으로 사용한다.

	매개변수:
		data: (N, ...) 형태의 key-value들이 들어 있는 Dictionary
		fold_i: 현재 사용할 test fold index
		fold_count: 전체 fold 개수
		fold_seed: generate_fold_index 함수에서 사용하는 무작위 시드
		stratify_key: 특정 key(예: "class")를 기준으로 계층화 분할을 수행할 때 사용.
					  None이면 일반 KFold로 진행한다.

	반환:
		split_dict: {
			"train": numpy array (train indices),
			"valid": numpy array (valid indices),
			"test":  numpy array (test indices)
		}
	"""
	folds_dict = generate_fold_index(
		data=data,
		n_fold=fold_count,
		seed=fold_seed,
		stratify_key=stratify_key
	)

	# test와 valid의 fold index 할당
	test_idx = folds_dict[fold_i]
	valid_idx = folds_dict[(fold_i + 1) % fold_count]

	# 나머지를 train으로 합침
	train_list = []
	for idx in range(fold_count):
		if idx not in [fold_i, (fold_i + 1) % fold_count]:
			train_list.append(folds_dict[idx])
	train_idx = np.concatenate(train_list, axis=0)

	split_dict = {
		"train": train_idx,
		"valid": valid_idx,
		"test":  test_idx
	}
	"""
	{
		'train': array([12, 15, ..., 91]), 
		'valid': array([ 0,  6, ..., 89]), 
		'test':  array([ 2,  8, ..., 94])
	}
	"""

	return split_dict



def generate_fold_index(
		data: dict,
		n_fold: int = 5,
		seed: int = 42,
		stratify_key: str = None
):
	"""
	data: (N, ...) 형태의 key-value들이 들어 있는 Dictionary
	n_fold: 생성할 Fold의 개수
	seed: 무작위 시드
	stratify_key: 특정 key(예: "class")를 기준으로 계층화 분할을 수행할 때 사용.
				  None이면 일반 KFold로 진행한다.

	반환:
		folds_dict: {
			"fold_0": numpy array(test_idx),
			"fold_1": numpy array(test_idx),
			...
		}
		형태의 Dictionary
	"""
	# sample 수 확인
	first_key = next(iter(data.keys()))
	num_samples = data[first_key].shape[0]
	indices = np.arange(num_samples)

	# 분할 결과를 담을 Dictionary
	folds_dict = {}

	# stratify_key가 주어졌을 때(StratifiedKFold 사용)
	if stratify_key is not None:
		y = data[stratify_key]
		if y.shape[0] != num_samples:
			raise ValueError(f"{stratify_key}의 첫 번째 차원이 전체 샘플 수와 맞지 않는다.")
		if len(y.shape) != 1:
			raise ValueError(f"{stratify_key}는 (N,) 형태의 1차원 라벨이어야 StratifiedKFold를 적용할 수 있다.")

		skf = StratifiedKFold(
			n_splits=n_fold,
			shuffle=True,
			random_state=seed
		)
		for i, (_, test_idx) in enumerate(skf.split(indices, y)):
			folds_dict[i] = test_idx

	# 일반 KFold 사용
	else:
		kf = KFold(
			n_splits=n_fold,
			shuffle=True,
			random_state=seed
		)
		for i, (_, test_idx) in enumerate(kf.split(indices)):
			folds_dict[i] = test_idx

	"""
	{
		0: array([ 2,  8, 11, 14, 17, 19, 22, 31, 38, 43, 54, 64, 70, 73, 79, 83, 86, 87, 94]), 
		1: array([ 0,  6,  9, 13, 25, 27, 30, 32, 40, 41, 50, 55, 56, 61, 67, 77, 80, 82, 89]), 
		2: array([12, 15, 16, 24, 33, 34, 36, 37, 46, 48, 49, 52, 62, 69, 71, 75, 81, 90, 93]), 
		3: array([ 3, 18, 20, 21, 23, 28, 35, 39, 42, 51, 59, 60, 63, 66, 68, 74, 84, 88, 92]), 
		4: array([ 1,  4,  5,  7, 10, 26, 29, 44, 45, 47, 53, 57, 58, 65, 72, 76, 78, 85, 91])
	}
	"""

	return folds_dict


# 함수 사용 예시
if __name__ == "__main__":
	# 예시 data 구성
	N = 10
	data_example = {
		"X": np.random.randn(N, 5),  # (10, 5)
		"class": np.random.randint(0, 3, size=(N,))  # 0,1,2 클래스 중 임의 할당
	}


	
