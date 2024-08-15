
def extract_testset(dataloader, n_sample):
	# test images
	batch = next(iter(dataloader))

	# batch가 tuple인지 확인하고, 각 요소의 처음 n_sample개를 반환
	if isinstance(batch, tuple):
		return tuple(item[:n_sample] for item in batch)
	else:
		return batch[:n_sample]