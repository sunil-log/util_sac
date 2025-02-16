
# -*- coding: utf-8 -*-
"""
Created on  Feb 16 2025

@author: sac
"""


import subprocess

def copy_to_clipboard(transcript):
	"""
	xclip을 사용해 클립보드에 주어진 문자열을 복사합니다.

	Args:
	    transcript (str): 클립보드로 복사할 문자열.
	"""
	subprocess.run(
		["xclip", "-selection", "clipboard"],
		input=transcript.encode('utf-8')
	)



if __name__ == "__main__":
	main()
