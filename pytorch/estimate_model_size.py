


def estimate_model_size(model):
	# 파라미터 개수 계산
	num_params = sum(p.numel() for p in model.parameters())
	print(f"파라미터 개수: {num_params:,}")

	# 메모리 사용량 계산 (bytes)
	mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
	mem_bufs = sum(buf.numel() * buf.element_size() for buf in model.buffers())
	mem_total = mem_params + mem_bufs
	print(f"메모리 사용량: {mem_total:,} bytes")
