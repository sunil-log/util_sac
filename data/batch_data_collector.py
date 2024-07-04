from collections import defaultdict


"""
mini-batch loop 직전에 선언하여 initialize 하고,
각 mini-batch 마다 계산된 loss 를 update 하면 다음을 수 행한다.
	
	total_loss += loss.item()
	num_batches += 1

mini-batch loop 이 끝나면, .average() 를 호출하여 각 loss 에 대해 num_batches 로 나눠주어야 한다.

- 이것은 메커니즘은 update 시에 받는 loss 가,
	mini-batch 안에서 sample 에 대한 평균이라는 것을 assume 한다.
"""


class batch_loss_collector:
	def __init__(self):
		self.losses = defaultdict(float)
		self.num_batches = 0

	def update(self, **loss_dict):
		for key, value in loss_dict.items():
			self.losses[key] += float(value)
		self.num_batches += 1

	def average(self):
		"""
		loss 가 mini-batch 안에서 sample 에 대한 평균이기 때문에,
		각 loss 에 대해서 num_batches 로 나눠주어야 한다.

		return {'train_loss': 0.5278644561767578, 'val_loss': 0.24387567241986594}
		"""
		return {key: value / self.num_batches for key, value in self.losses.items()}



if __name__ == '__main__':
	tracker = batch_loss_collector()
	for batch in range(10):
		tracker.update(
			train_loss=torch.tensor(0.5),
			val_loss=torch.tensor(0.3)
		)

	average_losses = tracker.average()
	print("Average losses:", average_losses)
