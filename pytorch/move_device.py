

def move_dict_tensors_to_device(d, device):
	"""
	이 함수는 dictionary 형태로 주어진 여러 Tensor를 지정된 device로 일괄 이동시켜줍니다.

	Args:
	    d (dict): Tensor를 값으로 가지는 dictionary.
	    device (torch.device): Tensor를 옮길 device.

	Returns:
	    dict: 각 Tensor가 device로 이동된 dictionary.
	"""
	return {k: v.to(device) for k, v in d.items()}
