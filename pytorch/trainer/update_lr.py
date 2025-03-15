
# -*- coding: utf-8 -*-
"""
Created on  Mar 15 2025

@author: sac
"""

def update_lr_with_dict(optimizer, epoch, lr_dict):
	"""
	lr_dict에 정의된 epoch가 되면 learning rate를 해당 값으로 갱신한다.
	예) lr_dict = {30: 1e-4, 80: 1e-5}

	호출 후, 현재(첫 번째 param group의) learning rate를 반환한다.
	"""
	# epoch가 lr_dict에 존재할 경우 learning rate 갱신
	if epoch in lr_dict:
		new_lr = lr_dict[epoch]
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		print(f"[Epoch {epoch}] Learning rate가 {new_lr}로 조정되었다.", flush=True)


def current_lr(optimizer):
	"""
	현재 optimizer의 learning rate를 반환한다.
	"""
	return {"lr": optimizer.param_groups[0]['lr']}

