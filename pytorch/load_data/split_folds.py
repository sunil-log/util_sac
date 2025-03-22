


import numpy as np
import torch

def make_k_fold_splits(
	data: dict,
	n_fold: int = 5,
	i_fold: int = 0,
	seed: int = 42,
	display_folds: bool = True  # fold 정보를 표시할지 여부
) -> dict:
	"""
	data: Dictionary 형태. key별 value가 (num_samples, ...) 형상을 가정한다.
	n_fold: 만들 Fold의 개수
	i_fold: Test용으로 사용할 Fold index (0 ~ n_fold-1)
	seed: 무작위 시드

	반환값:
		{
			'train': dict,
			'valid': dict,
			'test': dict
		}
	형태의 Dictionary. 각각의 key/value도 원본 data 구조를 유지하며
	Fold에 맞게 axis=0을 슬라이싱하여 분배한다.
	"""

	# (1) 무작위 시드 설정
	np.random.seed(seed)

	# (2) sample 수 확인
	# 임의로 data 안의 첫 번째 key의 shape[0]을 기준으로 사용
	# (모든 key가 동일한 axis=0 크기를 가진다고 가정)
	first_key = list(data.keys())[0]
	num_samples = data[first_key].shape[0]

	# (3) index 배열 생성 후 셔플
	indices = np.arange(num_samples)
	np.random.shuffle(indices)

	# (4) n_fold로 분할
	folds = np.array_split(indices, n_fold)

	# (5) Test와 Valid, Train에 해당하는 fold index 지정
	test_fold_idx = i_fold
	valid_fold_idx = (i_fold + 1) % n_fold

	test_idx = folds[test_fold_idx]
	valid_idx = folds[valid_fold_idx]

	# 나머지는 Train
	train_idx = []
	for f_idx in range(n_fold):
		if f_idx not in [test_fold_idx, valid_fold_idx]:
			train_idx.extend(folds[f_idx])
	train_idx = np.array(train_idx)


	# display_folds: fold 정보 표시 여부
	if display_folds:
		print(f"==== Fold Information ====")
		print(f"> Total samples: {num_samples}")
		for idx in [train_idx, valid_idx, test_idx]:
			print(f"> Fold shape: {idx.shape}, samples: {idx[:10]}")


	# (6) Dictionary 형태로 분할 데이터 구성
	data_train = {}
	data_valid = {}
	data_test = {}

	for key, value in data.items():
		# torch.Tensor, np.ndarray 모두 동일 로직
		data_train[key] = value[train_idx]
		data_valid[key] = value[valid_idx]
		data_test[key]  = value[test_idx]

	# (7) 반환
	return {
		'train': data_train,
		'valid': data_valid,
		'test': data_test
	}
