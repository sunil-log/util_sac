import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def generate_folds(
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
				  None이면 일반 KFold.

	반환:
		folds: 리스트 형태로, 각 원소는 (train_idx, test_idx) 쌍.
			   예: folds[i] = (train_idx, test_idx)
			   - train_idx, test_idx는 numpy array 형태
	"""
	# sample 수 확인
	first_key = next(iter(data.keys()))
	num_samples = data[first_key].shape[0]
	indices = np.arange(num_samples)



	# stratify_key가 주어졌을 때
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
		folds = list(skf.split(indices, y))
	else:
		kf = KFold(
			n_splits=n_fold,
			shuffle=True,
			random_state=seed
		)
		folds = list(kf.split(indices))

	return folds


# 함수 사용 예시
if __name__ == "__main__":
	# 예시 data 구성
	N = 10
	data_example = {
		"X": np.random.randn(N, 5),  # (10, 5)
		"class": np.random.randint(0, 3, size=(N,))  # 0,1,2 클래스 중 임의 할당
	}


	
