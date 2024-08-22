

"""
Envoronment
	pip install openai
	pip install --upgrade langchain


# make an instance
mygpt = ChatGPT()

def run_LLM(prompt):
	# return 	mygpt.get_completion(prompt, model="gpt-4-0125-preview")
	# return mygpt.get_completion(prompt, model="gpt-3.5-turbo-0125")
	return mygpt.get_completion(prompt, model="gpt-4o-2024-05-13")
"""


import os
from openai import OpenAI


import json


class ChatGPT:
	def __init__(self, key_file=None):
		"""
		ChatGPT 클래스를 초기화합니다.

		매개변수:
			key_file (str): OpenAI API 키가 포함된 파일의 경로입니다.
			                기본값은 현재 스크립트와 동일한 디렉토리의 'key_openai.json'입니다.
		"""
		if key_file is None:
			# 현재 스크립트의 절대 경로를 구합니다.
			current_dir = os.path.dirname(os.path.abspath(__file__))
			key_file = os.path.join(current_dir, 'key_openai.json')

		with open(key_file, 'r') as file:
			data = json.load(file)
			self.client = OpenAI(api_key=data["key"])


	def get_completion(self, prompt, model="gpt-3.5-turbo"):
		"""
		GPT 모델로부터 완성된 텍스트를 얻습니다.

		매개변수:
			prompt (str): 사용자의 입력 프롬프트입니다.
			model (str): GPT 모델의 ID입니다. 기본값은 'gpt-3.5-turbo'입니다.

		반환값:
			str: GPT 모델로부터의 응답 텍스트입니다.
		"""
		messages = [{"role": "user", "content": prompt}]
		response = self.client.chat.completions.create(model=model, messages=messages, temperature=0.1)
		return response.choices[0].message.content


