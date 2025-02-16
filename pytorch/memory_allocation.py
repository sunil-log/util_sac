
import torch


def estimate_model_size(model):
	# 파라미터 개수 계산
	num_params = sum(p.numel() for p in model.parameters())
	print(f"파라미터 개수: {num_params:,}", flush=True)

	# 메모리 사용량 계산 (bytes)
	mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
	mem_bufs = sum(buf.numel() * buf.element_size() for buf in model.buffers())
	mem_total = mem_params + mem_bufs
	print(f"메모리 사용량: {mem_total:,} bytes", flush=True)


def print_vram_usage(device='cuda'):
	"""
	지정된 device(GPU)의 현재 VRAM 사용량을 출력하는 함수입니다.
	PyTorch에서 제공하는 memory_allocated, memory_reserved 값을 확인하여
	실제 할당량(allocated)과 예약량(reserved)를 비교할 수 있습니다.

	Args:
		device (str): 확인할 디바이스, 기본값은 'cuda'입니다.
	"""
	# 정확한 측정을 위해서는 필요에 따라 torch.cuda.synchronize() 호출 가능
	allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB 단위
	reserved  = torch.cuda.memory_reserved(device)  / (1024**2)  # MB 단위

	print(f"[{device}] 현재 할당량(allocated): {allocated:.2f} MB", flush=True)
	print(f"[{device}] 현재 예약량(reserved) : {reserved:.2f} MB", flush=True)
