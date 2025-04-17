
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from util_sac.image_processing.reduce_palette import reduce_palette
from util_sac.pytorch.dataloader.to_tensor_device import dict_to_tensors


class ECG200Dataset(Dataset):
	"""
	UCR ECG200 데이터를 담는 Custom Dataset 클래스이다.
	__getitem__ 호출 시 {'trials': ..., 'label': ...} 형태의 dict를 반환한다.
	"""
	def __init__(self, data, labels):
		"""
		파라미터:
			trials (np.ndarray): shape = (N, T, 1)
			labels (np.ndarray): shape = (N,)
		"""
		super().__init__()
		# NumPy Array -> PyTorch Tensor
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return {
			'ecg': self.data[idx],
			'label': self.labels[idx]
		}


def process_ECG200_df(df: pd.DataFrame):
	"""
	이미 메모리에 로드된 ECG200 DataFrame(df)을 입력받아,
	(N, T, 1) 형태의 시계열 데이터와 0~(L-1) 범위로 변환된 정수 라벨을 반환한다.

	파라미터:
		df (pd.DataFrame): 0번째 컬럼이 라벨, 1~96번째 컬럼이 시계열인 DataFrame

	반환값:
		trials (np.ndarray): shape = (N, T, 1)
		labels (np.ndarray): shape = (N,)
	"""
	# 1) 라벨과 시계열 데이터 분리
	original_labels = df.iloc[:, 0].values  # 0번째 컬럼: 라벨
	data = df.iloc[:, 1:].values.astype(np.float64)  # 1~끝 컬럼: 시계열

	# 2) 라벨 매핑 (예: -1, 1 등을 0, 1 등으로 변환)
	unique_labels = np.unique(original_labels)
	label_map = {label: idx for idx, label in enumerate(unique_labels)}
	labels = np.array([label_map[label] for label in original_labels], dtype=int)

	# 3) Time Series 마지막 차원 확장 -> (N, T) -> (N, T, 1)
	data = data[..., np.newaxis]

	return data, labels


def load_ECG200(fn_train, fn_test, batch_size=32, shuffle=True):
	"""
	UCR ECG200 데이터셋만 로드하는 최소 예시 코드이다.
	TRAIN, TEST tsv 파일을 읽어 첫 번째 컬럼을 라벨로, 나머지를 시계열로 분리하고
	라벨을 0~(L-1) 범위로 변환한다.
	이후 마지막 차원을 1로 확장하여 (N, T, 1) 형태로 반환한다.
	"""

	# 파일 읽기
	train_df = pd.read_csv(fn_train, sep='\t', header=None)
	test_df = pd.read_csv(fn_test, sep='\t', header=None)
	"""
	    0         1         2         3   ...        93        94        95        96
	0   -1  0.502055  0.542163  0.722383  ...  0.708585  0.705011  0.713815  0.433765
	1    1  0.147647  0.804668  0.367771  ...  1.736054  0.036857 -1.265074 -0.208024
	2   -1  0.316646  0.243199  0.370471  ...  0.812345  0.748848  0.818042  0.539347
	3   -1  1.168874  2.075901  1.760141  ...  1.893512  1.256949  0.800407  0.731540
	4    1  0.648658  0.752026  2.636231  ... -0.304704 -0.454556  0.314590  0.582190
	..  ..       ...       ...       ...  ...       ...       ...       ...       ...
	95   1  0.581277  0.876188  1.042767  ...  0.762357  0.501373 -0.333336 -0.524546
	96  -1  2.689017  2.708703  2.008381  ... -0.243141 -0.119710  0.124042  0.273463
	97  -1  0.197677  0.455417  0.973110  ...  0.331451 -0.120006  0.042423  0.343293
	98   1  0.179500  1.038409  1.946421  ...  0.444768  0.151050  0.193378  0.451709
	99   1  0.073124  0.776054  2.181336  ... -0.447443 -0.066689 -0.178448 -0.256052
	"""

	# df to torch.Tensor
	train_data, train_labels = process_ECG200_df(train_df)
	test_data, test_labels = process_ECG200_df(test_df)
	d = {"train_data": train_data, "train_labels": train_labels,
	     "test_data": test_data, "test_labels": test_labels}
	d = dict_to_tensors(d)
	print_array_info(d)
	"""
	print_array_info(d)
	train_data PyTorch Tensor       (100, 96, 1)                37.50 KB torch.float32
	train_labels PyTorch Tensor       (100,)                      800.00 B torch.int64
	test_data  PyTorch Tensor       (100, 96, 1)                37.50 KB torch.float32
	test_labels PyTorch Tensor       (100,)                      800.00 B torch.int64
	"""

	# TensorDataset 생성
	train_dataset = ECG200Dataset(d["train_data"], d["train_labels"])
	test_dataset = ECG200Dataset(d["test_data"], d["test_labels"])

	# DataLoader 생성
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return {'train': train_loader, 'test': test_loader}


def plot_random_test_samples(
		test_data, test_labels,
		n_samples=5, alpha=0.3,
		save_path="test_samples_plot.png"
):
	"""
	test_data 에서 무작위로 n_samples 개 선택하여 시계열을 그려 저장합니다.
	각 시계열은 label 값(0,1)에 따라 다른 색상(red, blue)으로 표시합니다.

	Parameters:
	-----------
	test_data : np.ndarray
		(n_instances, n_timestamps, n_features) 형태의 Test 데이터
	test_labels : np.ndarray
		(n_instances,) 형태의 라벨 정보
	n_samples : int, optional
		무작위로 선택할 시계열 샘플의 개수 (기본값: 5)
	alpha : float, optional
		그래프 투명도 (기본값: 0.3)
	save_path : str, optional
		저장될 파일 경로 (기본값: "test_samples_plot.png")


    # 함수 사용 예시
    plot_random_test_samples(
        test_data,
        test_labels,
        n_samples=5,
        alpha=0.3,
        save_path="ecg200_test_samples.png"
    )
	"""

	# 무작위 시드 설정(필요시)
	np.random.seed(44)

	# test_data에서 임의로 n_samples개의 인덱스 선택
	assert len(test_data) >= n_samples, "n_samples should be less than or equal to the number of test instances"
	idxs = np.random.choice(len(test_data), size=n_samples, replace=False)

	# 그림 사이즈 설정(원하는 크기로 조절)
	plt.close()
	plt.figure(figsize=(16, 8))

	for i in idxs:
		X = test_data[i, :, 0]  # (n_timestamps,) 형태로 만들기
		label = test_labels[i]

		# label=0이면 빨강, 1이면 파랑
		color = 'red' if label == 0 else 'blue'
		plt.plot(X, color=color, alpha=alpha, label=f'Label {label}')

	plt.title(f"Random {n_samples} Test Instances from ECG200")
	plt.xlabel("Time")
	plt.ylabel("Value")

	# 라벨 중복 표시가 많아질 경우를 방지하기 위해 별도의 범례 생성
	# (아래와 같이 빈 라인으로 범례만 표시)
	# plt.plot([], [], color='red', label='Label 0')
	# plt.plot([], [], color='blue', label='Label 1')
	# plt.legend()

	# 그림 저장
	plt.tight_layout()
	img = reduce_palette(plt.gcf(), 32)
	img.save(save_path)
