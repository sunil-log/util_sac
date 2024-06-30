import numpy as np
from collections import defaultdict

class metric_tracker:
	"""
	지정된 메트릭스의 데이터를 동적으로 추적하고 분석하는 클래스입니다.
	이 클래스는 여러 메트릭스의 시계열 데이터를 저장하고, 최신 값, 전체 데이터,
	평균 등을 계산하는 기능을 제공합니다. 메트릭은 동적으로 추가할 수 있습니다.

	Attributes:
		data (defaultdict): 각 메트릭스의 데이터를 저장하는 딕셔너리

	Example:
		# FlexibleMetricTracker 인스턴스 생성
		tracker = FlexibleMetricTracker()

		# 데이터 업데이트 (메트릭이 자동으로 추가됨)
		tracker.update(accuracy=0.85, loss=0.35)
		tracker.update(accuracy=0.87, loss=0.32, f1_score=0.86)

		# 최신 값 조회
		latest_accuracy = tracker.get_latest('accuracy')
		print(f"최신 정확도: {latest_accuracy}")  # 출력: 최신 정확도: 0.87

		# 전체 데이터 조회
		all_loss = tracker.get_all('loss')
		print(f"모든 손실 데이터: {all_loss}")  # 출력: 모든 손실 데이터: [0.35, 0.32]

		# 현재 추적 중인 모든 메트릭 출력
		print(f"추적 중인 메트릭: {tracker.get_metrics()}")
	"""

	def __init__(self):
		self.data = defaultdict(list)

	def update(self, **kwargs):
		for metric, value in kwargs.items():
			self.data[metric].append(value)

	def get_latest(self, metric):
		if metric in self.data and self.data[metric]:
			return self.data[metric][-1]
		return None

	def get_all(self, metric):
		return self.data.get(metric, [])

	def __getitem__(self, metric):
		return self.get_all(metric)

	def get_metrics(self):
		return list(self.data.keys())

	def get_average(self, metric):
		values = self.get_all(metric)
		return np.mean(values) if values else None

	def print_latest(self):
		output = ", ".join(
			f"{metric}: {self.get_latest(metric) if self.get_latest(metric) is not None else '데이터 없음'}"
			for metric in self.get_metrics()
		)
		print(output, flush=True)

	def remove_metric(self, metric):
		if metric in self.data:
			del self.data[metric]

	def clear_data(self, metric=None):
		if metric:
			self.data[metric].clear()
		else:
			self.data.clear()