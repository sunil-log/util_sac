
# -*- coding: utf-8 -*-
"""
Created on  Apr 09 2025

@author: sac
"""


def deep_update(d, u):
	"""
	본 함수는 두 개의 dict를 재귀적으로 deep merge 한다. d와 u 모두 dict인 하위 요소에 대해서는
	d의 내용을 u의 내용으로 갱신한다. 덮어쓰기가 필요한 동일 key는 u 쪽 값으로 overwrite한다.

	예시:
		>>> d = {
		...	 "model": {"input_dim": 784, "hidden_dim": 256},
		...	 "data": {"path": "./data"}
		... }
		>>> u = {
		...	 "model": {"hidden_dim": 512},
		...	 "data": {"batch_size": 32}
		... }
		>>> updated_d = deep_update(d, u)
		>>> print(updated_d)
		{
			"model": {"input_dim": 784, "hidden_dim": 512},
			"data": {"path": "./data", "batch_size": 32}
		}

	Args:
		d (dict): deep merge의 대상이 되는 dict이다. 함수 호출 후 in-place로 갱신된다.
		u (dict): d에 반영할 key와 value를 가진 dict이다.

	Returns:
		dict: d에 u의 내용이 반영되어 재귀적으로 merge가 완료된 최종 dict이다.
	"""
	for k, v in u.items():
		if isinstance(v, dict) and k in d and isinstance(d[k], dict):
			deep_update(d[k], v)
		else:
			d[k] = v
	return d
