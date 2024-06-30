import torch
from collections import defaultdict
from typing import Callable, Dict, Any


class Metric:
	"""
	개별 메트릭을 저장하고 계산하는 클래스입니다.

	사용 예제:
	```python
	def update_loss(x):
		return x

	def compute_avg_loss(total, count):
		return total / count

	loss_metric = Metric('loss', update_loss, compute_avg_loss)
	loss_metric.update(2.5)
	loss_metric.update(3.0)
	avg_loss = loss_metric.compute()  # 결과: 2.75
	```
	"""

	def __init__(self, name: str, update_func: Callable, compute_func: Callable):
		self.name = name
		self.update_func = update_func
		self.compute_func = compute_func
		self.reset()

	def reset(self):
		self.value = 0
		self.count = 0

	def update(self, *args, **kwargs):
		value = self.update_func(*args, **kwargs)
		self.value += value
		self.count += 1

	def compute(self):
		if self.count == 0:
			return 0
		return self.compute_func(self.value, self.count)


class TrainingMetrics:
	"""
	여러 메트릭을 관리하고 계산하는 클래스입니다.

	사용 예제:
	```python
	from sklearn.metrics import accuracy_score, f1_score

	metrics = TrainingMetrics()

	# 기본 메트릭 추가
	metrics.add_metric('loss', lambda x: x, lambda total, count: total / count)

	# sklearn 메트릭 추가
	metrics.add_sklearn_metric('accuracy', accuracy_score)
	metrics.add_sklearn_metric('f1_score', f1_score)

	# 훈련 루프에서 메트릭 업데이트
	for epoch in range(num_epochs):
		for batch in dataloader:
			# ... 모델 훈련 코드 ...
			metrics.update(
				loss=loss.item(),
				logits=logits,
				labels=labels
			)
			# 이 때

		# 에포크 종료 후 결과 계산
		results = metrics.compute()
		print(f"Epoch {epoch+1} results:", results)
		metrics.reset()  # 다음 에포크를 위해 리셋
	```
	"""

	def __init__(self):
		self.metrics: Dict[str, Metric] = {}
		self.predictions = []
		self.targets = []

	def add_metric(self, name: str, update_func: Callable, compute_func: Callable):
		"""
		새로운 메트릭을 추가합니다.

		Args:
		    name (str): 메트릭의 이름
		    update_func (Callable): 메트릭 값을 업데이트하는 함수.
		                            배치마다 호출되며, 현재 배치의 메트릭 값을 반환해야 합니다.
		    compute_func (Callable): 최종 메트릭 값을 계산하는 함수.
		                             누적된 총 값과 업데이트 횟수를 인자로 받아 최종 값을 계산합니다.

		사용 예제:
		```python
		metrics = TrainingMetrics()
		metrics.add_metric('loss',
		                   update_func=lambda x: x,
		                   compute_func=lambda total, count: total / count)
		```
		"""
		self.metrics[name] = Metric(name, update_func, compute_func)

	def reset(self):
		for metric in self.metrics.values():
			metric.reset()
		self.predictions = []
		self.targets = []

	def update(self, **kwargs):
		for name, value in kwargs.items():
			if name in self.metrics:
				self.metrics[name].update(value)
		if 'logits' in kwargs and 'labels' in kwargs:
			self.predictions.extend(kwargs['logits'].argmax(dim=1).tolist())
			self.targets.extend(kwargs['labels'].tolist())

	def compute(self):
		results = {}
		for name, metric in self.metrics.items():
			results[name] = metric.compute()
		return results

	def add_sklearn_metric(self, name: str, sklearn_metric: Callable):
		"""
		scikit-learn 스타일의 메트릭을 추가합니다.

		Args:
		    name (str): 메트릭의 이름
		    sklearn_metric (Callable): scikit-learn의 메트릭 함수
		                               (예: sklearn.metrics.accuracy_score)

		이 메서드는 내부적으로 예측값과 실제값을 저장하고,
		compute 시점에 sklearn 메트릭 함수를 사용하여 최종 값을 계산합니다.

		사용 예제:
		```python
		from sklearn.metrics import accuracy_score
		metrics = TrainingMetrics()
		metrics.add_sklearn_metric('accuracy', accuracy_score)
		```
		"""
		def update_func(*args, **kwargs):
			return 0  # No-op, we'll compute this at the end

		def compute_func(*args, **kwargs):
			return sklearn_metric(self.targets, self.predictions)

		self.add_metric(name, update_func, compute_func)

# 나머지 코드는 그대로 유지...