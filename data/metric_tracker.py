
import numpy as np
from collections import defaultdict

class MetricsTracker:
	"""
	지정된 메트릭스의 데이터를 추적하고 분석하는 클래스입니다.

	이 클래스는 여러 메트릭스의 시계열 데이터를 저장하고, 최신 값, 전체 데이터,
	평균 등을 계산하는 기능을 제공합니다.

	Attributes:
		metrics (list): 추적할 메트릭스의 리스트
		data (defaultdict): 각 메트릭스의 데이터를 저장하는 딕셔너리

	Example:
		# MetricsTracker 인스턴스 생성
		tracker = MetricsTracker(['accuracy', 'loss'])

		# 데이터 업데이트
		tracker.update(accuracy=0.85, loss=0.35)
		tracker.update(accuracy=0.87, loss=0.32)

		# 최신 값 조회
		latest_accuracy = tracker.get_latest('accuracy')
		print(f"최신 정확도: {latest_accuracy}")  # 출력: 최신 정확도: 0.87

		# 전체 데이터 조회
		all_loss = tracker.get_all('loss')
		print(f"모든 손실 데이터: {all_loss}")  # 출력: 모든 손실 데이터: [0.35, 0.32]
	"""

	def __init__(self, metrics):
		self.metrics = metrics
		self.data = defaultdict(list)

	def update(self, **kwargs):
		for metric, value in kwargs.items():
			if metric in self.metrics:
				self.data[metric].append(value)

	def get_latest(self, metric):
		if metric in self.data:
			return self.data[metric][-1]
		return None

	def get_all(self, metric):
		return self.data.get(metric, [])

	def __getitem__(self, metric):
		return self.get_all(metric)

	def print_latest(self):
		for metric in self.metrics:
			latest = self.get_latest(metric)
			if latest is not None:
				print(f"{metric}: {latest}", flush=True)
			else:
				print(f"{metric}: 데이터 없음", flush=True)
