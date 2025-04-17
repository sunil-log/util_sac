
# -*- coding: utf-8 -*-
"""
Created on  Apr 07 2025

@author: sac
"""


import numpy as np
import pandas as pd
import torch

import util_sac.pytorch.dataloader as dlf
from util_sac.pandas.print_df import print_partial_markdown
from util_sac.pytorch.data.print_array import print_array_info
from util_sac.pytorch.dataloader.string_to_int import apply_or_create_map
from util_sac.pytorch.dataloader.int_to_onehot import one_hot_encode

from typing import Dict, Any

from torch.nn import functional as F



def create_iat_embeddings_repeated_rt(data: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Implicit Association Test (IAT) 단어와 반응 시간 데이터를 결합하여 임베딩 생성

	이 함수는 입력 딕셔너리 `data`에서 IAT 과제 관련 데이터인
	'iat_word' (단어 특징 벡터 배열)와 'iat_rt' (반응 시간 배열)를 사용합니다.
	반응 시간(rt) 데이터를 단어 특징 벡터의 마지막 차원 수(4)만큼 반복시킨 후,
	단어 특징 벡터와 결합(concatenate)하여 새로운 임베딩 벡터를 생성합니다.

	결과적으로 각 시행(trial)의 각 단계(time step)마다 4개의 단어 특징과
	4번 반복된 반응 시간 값이 결합되어 총 8차원의 특징 벡터를 갖는
	새로운 임베딩 배열('iat_emb_1', 'iat_emb_2')을 생성하여
	원본 딕셔너리 `data`에 추가하고 반환합니다.

	Args:
		data (Dict[str, Any]): 다음 키를 포함해야 하는 딕셔너리:
			- "iat_word": np.ndarray, 형태 (N, 2, 40, 4).
						  N은 참가자 수, 2는 조건 쌍(예: 긍정단어+자기 vs 부정단어+타인),
						  40은 시퀀스 길이(단어 제시/반응 단계 수),
						  4는 각 단어의 특징 벡터 차원.
			- "iat_rt": np.ndarray, 형태 (N, 2, 40).
						각 단어/자극 제시에 대한 반응 시간(ms 등).

	Returns:
		Dict[str, Any]: 입력 딕셔너리 `data`에 다음 키가 추가된 상태로 반환됨:
			- "iat_emb_1": np.ndarray, 형태 (N, 40, 8).
						   첫 번째 조건 쌍에 대한 결합 임베딩.
						   (4개의 단어 특징 + 4번 반복된 반응 시간)
			- "iat_emb_2": np.ndarray, 형태 (N, 40, 8).
						   두 번째 조건 쌍에 대한 결합 임베딩.
						   (4개의 단어 특징 + 4번 반복된 반응 시간)

	Side Effects:
		입력된 `data` 딕셔너리에 "iat_emb_1"와 "iat_emb_2" 키가 추가되어
		원본 딕셔너리가 변경됩니다.
	"""
	# 1. 데이터 추출
	iat_word = data["iat_word"]  # Shape: (N, 2, 40, 4)
	iat_rt = data["iat_rt"]	  # Shape: (N, 2, 40)

	n_dim = iat_word.shape[-1]  # 단어 특징 벡터 차원 (4)

	# 2. 반응 시간(rt) 차원 확장 및 반복
	# 마지막에 특징 차원 추가: (N, 2, 40) -> (N, 2, 40, 1)
	iat_rt_expanded = iat_rt[..., np.newaxis]
	# 새로 추가된 마지막 차원(axis=3)을 따라 값을 4번 반복: (N, 2, 40, 1) -> (N, 2, 40, 4)
	iat_rt_repeated = np.repeat(iat_rt_expanded, repeats=n_dim, axis=3)

	# 3. 단어 특징과 반복된 반응 시간 결합
	# 마지막 차원(axis=3)을 기준으로 결합: (N, 2, 40, 4) + (N, 2, 40, 4) -> (N, 2, 40, 8)
	data["iat_emb"] = np.concatenate([iat_word, iat_rt_repeated], axis=3)



	# del iat_word, iat_rt
	del data["iat_word"]  # iat_word는 임베딩 생성에 사용되므로 삭제
	del data["iat_rt"]    # iat_rt는 임베딩 생성에 사용되므로 삭제

	# 변경된 딕셔너리 반환
	return data


def process_iat_rt(data, log_transform=True, threshold=3.0, eps=1e-3):
	"""
	목적:
	  - IAT Reaction Time(RT)와 error 정보를 바탕으로, 필요 시 log transform을 수행하고
	    z-score를 구해 Outlier를 제거한다. Outlier나 error 구간은 valid_mask를 통해 마스킹하고,
	    최종적으로 RT 배열을 가공하여 data에 갱신한다.

	파라미터:
	  - data (dict):
	      * data["iat_rt"]: (n_subject, n_session, n_trial) 형태의 RT 배열
	      * data["error"]: (n_subject, n_session, n_trial) 형태의 error 배열 (1이면 error)
	  - log_transform (bool, 기본값 True):
	      * True일 경우 RT에 log(RT + eps)를 적용한다
	  - threshold (float, 기본값 3.0):
	      * Outlier 판정을 위한 z-score 컷오프
	  - eps (float, 기본값 1e-3):
	      * RT가 0 근처일 때 log(0) 발생을 방지하기 위해 더하는 상수

	처리 과정:
	  1. error=1인 구간을 valid_mask에서 제외(0으로 설정)한다.
	  2. (옵션) log_transform이 True이면, RT에 log(RT + eps)를 적용한다.
	  3. z-score를 계산하기 위해 피험자별로 (valid_mask==1)인 값만 골라 평균과 표준편차를 구한다.
	     - 표준편차가 매우 작은 경우(1e-12 이하)나 유효값이 부족하면 오류 메시지를 출력하고 종료한다.
	  4. z-score로 변환한 뒤, threshold를 초과하는 Outlier 구간을 valid_mask에서 0으로 마스킹한다.
	  5. valid_mask에서 0인 구간(Outlier+error)은 0으로 셋팅하고, 최종 z-score 값을 data["iat_rt"]에 저장한다.
	  6. 마지막으로 valid_mask를 data["iat_valid_mask"]에 추가한다.

	반환:
	  - data (dict):
	      * data["iat_rt"]: z-score로 변환된 RT 배열 (Outlier 및 error 구간은 0)
	      * data["iat_valid_mask"]: Outlier와 error가 마스킹된 배열
	"""

	rt = data["iat_rt"]	  # shape: (n_subject, n_session, n_trial)
	error = data["error"]	# shape: (n_subject, n_session, n_trial)

	# 새 마스크 배열 생성 (int 타입 권장)
	valid_mask = np.ones_like(rt, dtype=np.int64)

	# 1. error=1인 구간 제외
	valid_mask[error == 1] = 0

	# 2. (옵션) log 변환: 필요하다면 rt에 직접 반영
	if log_transform:
		rt = np.log(rt + eps)

	# --- z-score 계산용 임시 배열 ---
	z_rt = np.zeros_like(rt)

	n_subject = rt.shape[0]
	for subj_idx in range(n_subject):
		rt_subj = rt[subj_idx]		 # (n_session, n_trial)
		mask_subj = valid_mask[subj_idx]

		valid_vals = rt_subj[mask_subj == 1]
		if len(valid_vals) < 2:
			print(f"피험자 {subj_idx}의 유효값이 부족하여 z-score 계산을 종료합니다.")
			exit()

		mean_rt = valid_vals.mean()
		std_rt = valid_vals.std()
		if std_rt < 1e-12:
			print(f"피험자 {subj_idx}의 표준편차가 너무 작아 계산을 종료합니다.")
			exit()

		z_subj = (rt_subj - mean_rt) / std_rt  # z-score
		z_rt[subj_idx] = z_subj

		# outlier 마스킹
		outliers = np.abs(z_subj) > threshold
		mask_subj[outliers] = 0

	# 3. outlier + error 구간을 0으로 설정
	z_rt[valid_mask == 0] = 0

	# 4. 최종적으로 data["iat_rt"]에 z_rt를 덮어쓴다
	data["iat_rt"] = z_rt
	data["iat_valid_mask"] = valid_mask

	return data


def process_iat_message(data, args):
	"""
	IAT(Implicit Association Test)에서 추출된 정보인 A와 B를 기반으로 특정 단어를 분류하고,
	A의 값(1 혹은 2)에 따라 최종 결과를 산출한다. A는 왼쪽(1)인지 오른쪽(2)인지를 나타내는 정수이고,
	B는 임의의 단어 목록이다. 이 중 args에서 지정된 'target_name'과 'base_name'을 인식하여
	0, 1, 2로 매핑한 뒤, A의 값에 따라 결과에 가중을 더해 최종 array를 만든다.

	:param data: IAT 처리 과정에서 얻어진 데이터 구조체이다.
	    - "listAns": A에 해당하며, 1(왼쪽) 또는 2(오른쪽)을 나타내는 값들이 있다.
	    - "StimRaw": B에 해당하며, 실제 문자열(단어) 목록이 들어 있다.
	:param args: IAT 분석에 필요한 추가 파라미터를 담고 있는 dictionary이다.
	    - "target_name": B 중에서 0으로 분류할 단어(예: 특정 후보 이름)에 해당한다.
	    - "base_name": B 중에서 1로 분류할 단어에 해당한다.
	:return: B에 대해 0, 1, 2로 기본 매핑한 뒤, B_map이 2(해당 없음)인 위치에 한해서
	         (A - 1)을 더한 값으로 구성된 최종 array이다.
	         - 0: B가 target_name일 때
	         - 1: B가 base_name일 때
	         - 2: 그 외의 단어이고 A가 1일 때
	         - 3: 그 외의 단어이고 A가 2일 때
	"""


	# extract 문재인, 안철수
	name1, name2 = args["targets"]

	if args["iat_use_name"]: # 안철수, 문재인 따로 -> 4개의 가능성

		# words
		A = data["listAns"]
		B = data["StimRaw"]
		B_map = (B == name1).astype(int) * 0 \
		        + (B == name2).astype(int) * 1 \
		        + ((B != name1) & (B != name2)).astype(int) * 2
		final = B_map + (B_map == 2) * (A-1)


	else: #  왼쪽-오른쪽 -> 2개의 가능성
		A = data["listAns"]
		final = (A-1)

	return final


def pre_processing_iat(data, args):

	"""
	IAT(Implicit Association Test) 관련 데이터를 전처리하는 함수입니다.

	이 함수는 원본 데이터셋을 받아 IAT 자극어 처리, 불필요 컬럼 제거,
	컬럼명 변경, 데이터 타입 변환(문자열 -> 정수, 원-핫 인코딩),
	타겟 변수 이진화, 특정 변수 정규화 (나이, IAT 반응 시간),
	인구통계학적 정보 통합, 그리고 최종 IAT 임베딩 생성을 포함한
	일련의 전처리 과정을 수행합니다.

	처리 과정:
	1.  `StimRaw`와 `listAns` 컬럼을 이용하여 IAT 자극어(`iat_word`)를 생성합니다.
		(`process_iat_message` 함수 호출)
	2.  `iat_word` 생성에 사용된 `StimRaw`, `listAns` 컬럼과,
		`error` 컬럼과 동일한 정보를 가진 `pressedKey` 컬럼을 삭제합니다.
	3.  타겟 변수인 `RealVote` 컬럼명을 `y`로 변경하고,
		IAT 반응 시간인 `rt` 컬럼명을 `iat_rt`로 변경합니다.
	4.  후속 처리 (텐서 변환 등)를 위해 지정된 필드들
		('path', 'residence', 'gender', 'y', 'W단어', 'W선택단어', 'W제시자극', 'WEvalWord')의
		데이터 타입을 문자열(str)에서 정수(int)로 변환합니다.
		이 과정에서 각 필드별 매핑(mapping) 정보를 생성하거나 기존 정보를 사용합니다.
		(`apply_or_create_map` 함수 사용)
	5.  'residence', 'gender', 'iat_word' 컬럼에 대해 원-핫 인코딩을 적용합니다.
		(`one_hot_encode` 함수 사용)
	6.  타겟 변수 `y`를 이진(binary) 형식으로 변환합니다. `args`에 지정된
		두 타겟 중 첫 번째 타겟(e.g., '문재인')은 1로, 나머지는 0으로 변환됩니다.
	7.  `age` 컬럼을 실수형(float)으로 변환합니다. `args["normalize"]["age"]` 설정이
		True인 경우, 나이 데이터를 표준화(평균 0, 표준편차 1)합니다.
	8.  처리된 'residence'(원-핫), 'gender'(원-핫), 'age'(정규화된 실수) 데이터를
		결합하여 인구통계학적 정보(`demographics`) 텐서(Tensor)를 생성합니다.
		생성 후 원본 'residence', 'gender', 'age' 컬럼은 삭제됩니다.
	9.  IAT 반응 시간(`iat_rt`)을 `args["normalize"]["iat"]` 설정에 따라
		정규화합니다. 예를 들어 'log-z' 방식이 지정된 경우, 로그 변환 후
		표준편차(`threshold`) 기반 이상치를 제거하고 Z-점수 정규화를 수행합니다.
		(`process_iat_rt` 함수 사용)
	10. `iat_valid_mask`에 정보가 반영된 `error` 컬럼을 삭제합니다.
	11. 최종적으로 처리된 `iat_word`(원-핫)와 `iat_rt`(정규화)를 결합하여
		IAT 임베딩(`iat_embedding`)을 생성합니다.
		(`create_iat_embeddings_repeated_rt` 함수 사용)

	Args:
		data (dict 또는 pandas.DataFrame): 전처리할 원본 데이터.
			다양한 초기 컬럼들('StimRaw', 'listAns', 'pressedKey', 'RealVote',
			'rt', 'path', 'residence', 'gender', 'age', 'W단어' 등)을 포함합니다.
		args (dict): 전처리 과정을 제어하는 설정값 딕셔너리.
			'db_dir', 'targets', 'normalize' (내부에 'age', 'iat' 설정 포함) 등의
			키를 포함할 수 있습니다.

	Returns:
		dict 또는 pandas.DataFrame: 전처리된 데이터.
			최종적으로 'path', 'W단어', 'W선택단어', 'W응답시간', 'W제시자극',
			'WDwelling', 'WEvalWord', 'WVisits', 'y', 'demographics',
			'iat_valid_mask', 'iat_emb' 등의 컬럼을 포함할 수 있습니다.
			(함수 마지막 주석 참고)
	"""


	"""
	path       NumPy Array          (80,)                       640.00 B int64
	error      NumPy Array          (80, 2, 40)                 50.00 KB int64
	listAns    NumPy Array          (80, 2, 40)                 50.00 KB int64
	pressedKey NumPy Array          (80, 2, 40)                 50.00 KB int64
	StimRaw    NumPy Array          (80, 2, 40)                 75.00 KB <U3
	W단어        NumPy Array          (80, 9, 16)                 90.00 KB int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	W제시자극      NumPy Array          (80, 9, 1)                   5.62 KB int64
	WDwelling  NumPy Array          (80, 9, 16)                 45.00 KB float32
	WEvalWord  NumPy Array          (80, 9, 16)                 90.00 KB int64
	WVisits    NumPy Array          (80, 9, 16)                 45.00 KB float32
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	iat_rt     NumPy Array          (80, 2, 40)                 25.00 KB float32
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	"""


	"""
	process_iat_message 
		"iat_use_name" == True 이면 
			후보1, 후보2, good, bad 를 각각 0, 1, 2, 3  바꿔준다.
		
		"iat_use_name" == False 이면
			good, bad 를 각각 0, 1 로 바꿔준다.
	"""
	data["iat_word"] = process_iat_message(data, args)
	del data["StimRaw"]         # StimRaw는 iat_word 를 만드는데 사용되므로 삭제한다.
	del data["listAns"]         # listAns는 iat_word 를 만드는데 사용되므로 삭제한다.
	del data["pressedKey"]      # pressedKey는 error 와 같아서 삭제한다.




	"""
	일부를 one-hot 으로 변환
	"""
	for key in ["iat_word"]:
		data[key] = one_hot_encode(data[key])


	"""
	IAT normalization
	"""
	norm_info = args["normalize"]["iat"]
	if norm_info["method"] == "log-z":
		data = process_iat_rt(
			data,
			log_transform=True,
			threshold=norm_info["sd"],
			eps=1.0,     # ms 라서 1정도 더해도 문제 없다.
		)
	elif norm_info["method"] == "z":
		data = process_iat_rt(
			data,
			log_transform=False,
			threshold=norm_info["sd"],
			eps=1.0,     # ms 라서 1정도 더해도 문제 없다.
		)


	# 불필요한 key 삭제
	del data["error"]          # error 는 iat_valid_mask 에 포함되어 있다.


	"""
	iat_word 와 iat_rt 를 concat 하여 iat_embedding 을 만든다.
	"""
	data = create_iat_embeddings_repeated_rt(data)

	"""
	path       NumPy Array          (81,)                       648.00 B int64
	W단어        NumPy Array          (81, 9, 16)                 91.12 KB int64
	W선택단어      NumPy Array          (81, 9, 1)                   5.70 KB int64
	W응답시간      NumPy Array          (81, 9, 1)                   2.85 KB float32
	W제시자극      NumPy Array          (81, 9, 1)                   5.70 KB int64
	WDwelling  NumPy Array          (81, 9, 16)                 45.56 KB float32
	WEvalWord  NumPy Array          (81, 9, 16)                 91.12 KB int64
	WVisits    NumPy Array          (81, 9, 16)                 45.56 KB float32
	y          NumPy Array          (81,)                       648.00 B int64
	demographics PyTorch Tensor       (81, 8)                      2.53 KB torch.float32
	iat_valid_mask NumPy Array          (81, 2, 40, 1)              50.62 KB int64
	iat_emb    NumPy Array          (81, 2, 40, 8)             202.50 KB float32
	"""

	return data



def pre_processing_wat(data, args):

	"""
	WDwelling 과 WEvalWord 을 사용한다.
	9개의 session 이 0, 1, 2, 은 [토끼, 호랑이, 뱀] 이고,
		3, 4, 5, 은 [문재인, 문재인, 문재인], 6, 7, 8 은 [안철수, 안철수, 안철수] 이다.

	따라서 [3, 4, 5] 를 하나로 묶고, [6, 7, 8] 을 하나로 묶는다.
	- (81, 9, 16) 중에 6개만 빼면 (81, 6, 16) 이 된다
	- 단어중 첫번째 것은 제시단어로 사용되지 않으므로 빼면 (81, 6, 15) 가 된다.
	- 6을 2개의 session 으로 보면  (81, s=2, trial=45) 이 된다.
	- WEvalWord 는 one-hot 으로 encoding 하면 (81, 2, 45, 3) 이 된다.

	"""

	"""
	path       NumPy Array          (80,)                       640.00 B int64
	W단어        NumPy Array          (80, 9, 16)                 90.00 KB int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	W제시자극      NumPy Array          (80, 9, 1)                   5.62 KB int64
	WDwelling  NumPy Array          (80, 9, 16)                 45.00 KB float32
	WEvalWord  NumPy Array          (80, 9, 16)                 90.00 KB int64
	WVisits    NumPy Array          (80, 9, 16)                 45.00 KB float32
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	iat_valid_mask NumPy Array          (80, 2, 40)                 50.00 KB int64
	iat_emb    NumPy Array          (80, 2, 40, 4)             100.00 KB float32
	"""

	# 필요없는 정보 제거
	del data["W단어"]          # 실재 단어는 사용되지 않아 제거한다.
	del data["W제시자극"]       # 제시자극은 문재인 [3:6], 안철수[6:9] 으로 고정되어 제거한다.
	del data["WVisits"]       # WVisits 은 사용되지 않아 제거한다.


	# extract words
	wat_word = data["WEvalWord"][:, 3:9, 1:]    # (81, 6, 15); `1:` 첫번째 단어는 제시 자극이라 제외


	# normalize time
	wat_tt = data["W응답시간"][:, 3:9]        # (81, 6, 1)
	wat_time = data["WDwelling"][:, 3:9]    # (81, 6, 16)
	wat_time = wat_time / wat_tt
	wat_time = wat_time[:, :, 1:]           # (81, 6, 15); `1:` 첫번째 단어는 제시 자극이라 제외


	# split by two sessions
	wat_word = wat_word.reshape(wat_word.shape[0], 2, 3, -1)  # (81, 2, 3, 15)
	wat_time = wat_time.reshape(wat_time.shape[0], 2, 3, -1)  # (81, 2, 3, 15)
	wat_word = wat_word.reshape(wat_word.shape[0], 2, -1)  # (81, 2, 45)
	wat_time = wat_time.reshape(wat_time.shape[0], 2, -1)  # (81, 2, 45)


	# 일부를 one-hot 으로 변환
	wat_word = one_hot_encode(wat_word)     # (81, 2, 45, 3)
	wat_time = wat_time[..., np.newaxis]    # (81, 2, 45, 1)


	# combine word and time as wat_emb
	n_dim = wat_word.shape[-1]  # 단어 특징 벡터 차원 (3)
	wat_time = np.repeat(wat_time, repeats=n_dim, axis=3)  # (81, session=2, 45, 3)
	wat_emb = np.concatenate([wat_word, wat_time], axis=3)  # (81, session=2, 45, 6)
	data["wat_emb"] = wat_emb



	# del WDwelling and WEvalWord
	del data["WDwelling"]  # WDwelling 은 wat_emb 를 만드는데 사용되므로 삭제한다.
	del data["WEvalWord"]  # WEvalWord 은 wat_emb 를 만드는데 사용되므로 삭제한다.
	del data["W응답시간"]    # W응답시간 은 wat_emb 를 만드는데 사용되므로 삭제한다.



	"""
	path       NumPy Array          (80,)                       640.00 B int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	iat_valid_mask NumPy Array          (80, 2, 40)                 50.00 KB int64
	iat_emb    NumPy Array          (80, 2, 40, 4)             100.00 KB float32
	wat_emb    NumPy Array          (80, 2, 45, 6)             168.75 KB float32
	"""

	return data



def data_pre_processing_wat_select(data, args):

	"""
	두 후보(Candidate 1: index 3-5, Candidate 2: index 6-8)별로 선택된 단어의
	category count를 계산하고 결합하여 feature vector를 생성한다.

	Process:
	1. 평가 데이터(`W선택단어평가`)에서 후보 1, 2에 해당하는 category 값(index 3-8)을 선택한다 (`(N, 6)`).
	2. 선택된 값을 one-hot encoding한다 (`(N, 6, num_classes)`).
	3. `view`와 `sum`을 사용해 각 후보별 category count를 계산하고 (`(N, 2, num_classes)`),
	   이를 `view`를 통해 하나의 tensor로 결합한다 (`(N, 2 * num_classes)`).

	Input: `data['W선택단어평가']`의 슬라이스 (`[:N, 3:9, 0]`).
	Output: `result` tensor, 각 샘플에 대한 [후보1 counts, 후보2 counts] 형태의
	        결합된 category count 벡터 (`(N, 2 * num_classes)`).
	"""
	num_samples = data["W선택단어평가"].shape[0]                # 81

	wat_select = data["W선택단어평가"][:, 3:9, 0]               # Shape: (81, 6)
	wat_select = one_hot_encode(wat_select)                  # Shape: (81, 6, 3)
	wat_select = wat_select.view(num_samples, 2, 3, -1)      # Shape: (81, m=2, s=3, d=3)
	wat_select = wat_select.sum(dim=2)                       # Shape: (81, m=2, d=3)
	wat_select = wat_select.view(num_samples, -1)            # Shape: (81, m=2*3)


	data["wat_select"] = wat_select  # (81, d=6)
	"""
	웃기게도 normalize 를 하지 않고 그냥 1, 2, 3 같은 값이 나오는게 점수가 더 안정적이다.
	sotfmax 도 없고, 3.0 으로 나누는것도 별로다.
	"""

	del data["W선택단어"]       # W선택단어평가는 wat_select 를 만드는데 사용되므로 삭제한다.
	del data["W선택단어평가"]    # W선택단어평가는 wat_select 를 만드는데 사용되므로 삭제한다.



	"""
	path       NumPy Array          (80,)                       640.00 B int64
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	y          NumPy Array          (80,)                       640.00 B int64
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	iat_valid_mask NumPy Array          (80, 2, 40)                 50.00 KB int64
	iat_emb    NumPy Array          (80, 2, 40, 4)             100.00 KB float32
	wat_emb    PyTorch Tensor       (80, 6)                      1.88 KB torch.float32
	wat_select PyTorch Tensor       (80, 6)                      1.88 KB torch.float32
	"""

	return data


def data_pre_processing_demographics(data, args):


	"""
	"""

	"""
	path       NumPy Array          (80,)                        8.75 KB <U28
	residence  NumPy Array          (80,)                       640.00 B <U2
	gender     NumPy Array          (80,)                       320.00 B <U1
	age        NumPy Array          (80,)                       640.00 B int64
	RealVote   NumPy Array          (80,)                       960.00 B <U3
	error      NumPy Array          (80, 2, 40)                 50.00 KB int64
	listAns    NumPy Array          (80, 2, 40)                 50.00 KB int64
	pressedKey NumPy Array          (80, 2, 40)                 50.00 KB int64
	rt         NumPy Array          (80, 2, 40)                 25.00 KB float32
	StimRaw    NumPy Array          (80, 2, 40)                 75.00 KB <U3
	W단어        NumPy Array          (80, 9, 16)                405.00 KB <U9
	W선택단어      NumPy Array          (80, 9, 1)                  25.31 KB <U9
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	W제시자극      NumPy Array          (80, 9, 1)                  22.50 KB <U8
	WDwelling  NumPy Array          (80, 9, 16)                 45.00 KB float32
	WEvalWord  NumPy Array          (80, 9, 16)                 90.00 KB <U2
	WVisits    NumPy Array          (80, 9, 16)                 45.00 KB float32
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB <U2
	"""



	"""
	label 이름 RealVote -> y 로 바꾸기
	"""
	data["y"] = data.pop("RealVote")
	data["iat_rt"] = data.pop("rt")


	"""
	tensor 로 바꾸기 위해서 str -> int 로 바꿔야 한다.
	"""
	field_list = ['path', 'residence', 'gender', 'y', 'W단어', 'W선택단어', 'W제시자극', 'WEvalWord', "W선택단어평가"]
	maps = {}
	for field in field_list:
		data[field], maps[field] = apply_or_create_map(
			data=data[field], map_name=field, root_dir=args["db_dir"])

	"""
	일부를 one-hot 으로 변환
	"""
	for key in ["residence", "gender"]:
		data[key] = one_hot_encode(data[key])

	"""
	Label(RealVote) 을 Binary (e.g. 문재인: 1, Others: 0) 로.
	"""
	name1, name2 = args["targets"]
	y_1 = maps['y'][name1]     # e.g. 문재인 0 -> 1
	data["y"] = np.where(data["y"] == y_1, 1, 0)

	"""
	age normalization
	"""
	data["age"] = data["age"].astype(np.float64)
	if args["normalize"]["age"]:
		maen_age = np.mean(data["age"])
		std_age = np.std(data["age"])
		data["age"] = (data["age"] - maen_age) / std_age


	"""
	demographic data 를 묶는다; (residence, gender, age) -> demographics
	"""
	age_numpy = data["age"][..., np.newaxis]
	age_numpy = np.repeat(age_numpy, 3, axis=1)  # (81,) -> (81, 3)

	residence_tensor = data["residence"]
	gender_tensor = data["gender"]
	age_tensor = torch.from_numpy(age_numpy).float()  # .float()으로 float32로 변환
	demographics_tensor = torch.cat((residence_tensor, gender_tensor, age_tensor), axis=1)
	data['demographics'] = demographics_tensor

	del data["residence"]
	del data["gender"]
	del data["age"]



	"""
	path       NumPy Array          (80,)                       640.00 B int64
	error      NumPy Array          (80, 2, 40)                 50.00 KB int64
	listAns    NumPy Array          (80, 2, 40)                 50.00 KB int64
	pressedKey NumPy Array          (80, 2, 40)                 50.00 KB int64
	StimRaw    NumPy Array          (80, 2, 40)                 75.00 KB <U3
	W단어        NumPy Array          (80, 9, 16)                 90.00 KB int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	W제시자극      NumPy Array          (80, 9, 1)                   5.62 KB int64
	WDwelling  NumPy Array          (80, 9, 16)                 45.00 KB float32
	WEvalWord  NumPy Array          (80, 9, 16)                 90.00 KB int64
	WVisits    NumPy Array          (80, 9, 16)                 45.00 KB float32
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	iat_rt     NumPy Array          (80, 2, 40)                 25.00 KB float32
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	"""

	return data




def load_data(args):

	# load data
	data = dict(np.load(args["data_npz"]))
	"""
	path       NumPy Array          (81,)                        8.86 KB <U28
	residence  NumPy Array          (81,)                       648.00 B <U2
	gender     NumPy Array          (81,)                       324.00 B <U1
	age        NumPy Array          (81,)                       648.00 B int64
	RealVote   NumPy Array          (81,)                       972.00 B <U3
	error      NumPy Array          (81, 2, 40)                 50.62 KB int64
	listAns    NumPy Array          (81, 2, 40)                 50.62 KB int64
	pressedKey NumPy Array          (81, 2, 40)                 50.62 KB int64
	rt         NumPy Array          (81, 2, 40)                 25.31 KB float32
	StimRaw    NumPy Array          (81, 2, 40)                 75.94 KB <U3
	W단어        NumPy Array          (81, 9, 16)                410.06 KB <U9
	W선택단어      NumPy Array          (81, 9, 1)                  25.63 KB <U9
	W응답시간      NumPy Array          (81, 9, 1)                   2.85 KB float32
	W제시자극      NumPy Array          (81, 9, 1)                  22.78 KB <U8
	WDwelling  NumPy Array          (81, 9, 16)                 45.56 KB float32
	WEvalWord  NumPy Array          (81, 9, 16)                 91.12 KB <U2
	WVisits    NumPy Array          (81, 9, 16)                 45.56 KB float32
	"""


	data = data_pre_processing_demographics(data, args)
	"""
	path       NumPy Array          (80,)                       640.00 B int64
	error      NumPy Array          (80, 2, 40)                 50.00 KB int64
	listAns    NumPy Array          (80, 2, 40)                 50.00 KB int64
	pressedKey NumPy Array          (80, 2, 40)                 50.00 KB int64
	StimRaw    NumPy Array          (80, 2, 40)                 75.00 KB <U3
	W단어        NumPy Array          (80, 9, 16)                 90.00 KB int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W응답시간      NumPy Array          (80, 9, 1)                   2.81 KB float32
	W제시자극      NumPy Array          (80, 9, 1)                   5.62 KB int64
	WDwelling  NumPy Array          (80, 9, 16)                 45.00 KB float32
	WEvalWord  NumPy Array          (80, 9, 16)                 90.00 KB int64
	WVisits    NumPy Array          (80, 9, 16)                 45.00 KB float32
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	iat_rt     NumPy Array          (80, 2, 40)                 25.00 KB float32
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	"""



	data = pre_processing_iat(data, args)
	# print_array_info(data)
	"""
	path       NumPy Array          (81,)                       648.00 B int64
	W단어        NumPy Array          (81, 9, 16)                 91.12 KB int64
	W선택단어      NumPy Array          (81, 9, 1)                   5.70 KB int64
	W응답시간      NumPy Array          (81, 9, 1)                   2.85 KB float32
	W제시자극      NumPy Array          (81, 9, 1)                   5.70 KB int64
	WDwelling  NumPy Array          (81, 9, 16)                 45.56 KB float32
	WEvalWord  NumPy Array          (81, 9, 16)                 91.12 KB int64
	WVisits    NumPy Array          (81, 9, 16)                 45.56 KB float32
	y          NumPy Array          (81,)                       648.00 B int64
	demographics PyTorch Tensor       (81, 8)                      2.53 KB torch.float32
	iat_valid_mask NumPy Array          (81, 2, 40, 1)              50.62 KB int64
	iat_emb    NumPy Array          (81, 2, 40, 8)             202.50 KB float32
	"""



	data = pre_processing_wat(data, args)
	"""
	path       NumPy Array          (80,)                       640.00 B int64
	W선택단어      NumPy Array          (80, 9, 1)                   5.62 KB int64
	W선택단어평가    NumPy Array          (80, 9, 1)                   5.62 KB int64
	y          NumPy Array          (80,)                       640.00 B int64
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	iat_valid_mask NumPy Array          (80, 2, 40)                 50.00 KB int64
	iat_emb    NumPy Array          (80, 2, 40, 4)             100.00 KB float32
	wat_emb    NumPy Array          (80, 2, 45, 6)             168.75 KB float32
	"""



	data = data_pre_processing_wat_select(data, args)
	"""
	y          NumPy Array          (80,)                       640.00 B int64
	demographics PyTorch Tensor       (80, 8)                      2.50 KB torch.float32
	iat_valid_mask NumPy Array          (80, 2, 40)                 50.00 KB int64
	iat_emb    NumPy Array          (80, 2, 40, 4)             100.00 KB float32
	wat_emb    NumPy Array          (80, 2, 45, 6)             168.75 KB float32
	wat_select PyTorch Tensor       (80, 6)                      1.88 KB torch.float32
	"""



	"""
	split data into train, valid, test
	"""
	data = dlf.split_data_into_train_valid_test(
		data=data,
		fold_i=args["fold"]["i"],
		fold_count=args["fold"]["count"],
		fold_seed=args["fold"]["seed"],
		stratify_key="y"
	)
	"""
	print_array_info(data)
	train      Other                <class 'dict'>                   N/A N/A       
	valid      Other                <class 'dict'>                   N/A N/A       
	test       Other                <class 'dict'>                   N/A N/A
	"""

	"""
	print_array_info(data["train"])
	"""


	df_count = dlf.label_distribution_table(data, label_col="y")
	print_partial_markdown(df_count)
	"""
	|    | Class   |   count_train |   percent_train |   count_valid |   percent_valid |   count_test |   percent_test |
	|---:|:--------|--------------:|----------------:|--------------:|----------------:|-------------:|---------------:|
	|  0 | 0       |            24 |              50 |             7 |           43.75 |            8 |        47.0588 |
	|  1 | 1       |            24 |              50 |             9 |           56.25 |            9 |        52.9412 |
	|  2 | Total   |            48 |             100 |            16 |          100    |           17 |       100      |
	"""



	dataloaders = dlf.create_dataloaders(data, batch_size=16, shuffle=True)
	"""
	train      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	valid      Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A       
	test       Other                <class 'torch.utils.data.dataloader.DataLoader'>             N/A N/A  
	"""

	"""
	batch = next(iter(dataloaders["train"]))
	print_array_info(batch)
		
	path            PyTorch Tensor       (16,)                       128.00 B torch.int64
	W단어            PyTorch Tensor       (16, 9, 16)                 18.00 KB torch.int64
	W선택단어         PyTorch Tensor       (16, 9, 1)                   1.12 KB torch.int64
	W응답시간         PyTorch Tensor       (16, 9, 1)                  576.00 B torch.float32
	W제시자극         PyTorch Tensor       (16, 9, 1)                   1.12 KB torch.int64
	WDwelling       PyTorch Tensor       (16, 9, 16)                  9.00 KB torch.float32
	WEvalWord       PyTorch Tensor       (16, 9, 16)                 18.00 KB torch.int64
	WVisits         PyTorch Tensor       (16, 9, 16)                  9.00 KB torch.float32
	y               PyTorch Tensor       (16,)                       128.00 B torch.int64
	demographics    PyTorch Tensor       (16, 8)                     512.00 B torch.float32
	iat_valid_mask  PyTorch Tensor       (16, 2, 40, 1)              10.00 KB torch.int64
	iat_emb         PyTorch Tensor       (16, 2, 40, 8)              40.00 KB torch.float32
	"""

	return dataloaders





