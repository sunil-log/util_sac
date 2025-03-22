
"""
cron 에서 사용할 경우, DISPLAY 환경변수가 없어서 오류가 발생하는 경우가 있습니다.
이럴 경우, 아래와 같이 DISPLAY 환경변수를 설정해주면 됩니다.
	# specify display
	# echo $DISPLAY
	export DISPLAY=:1
"""
import subprocess
import pandas as pd


def convert_dict_to_str(my_dict):
	"""
	딕셔너리를 문자열로 변환하는 함수입니다.

	Args:
	my_dict (dict_module): 변환할 딕셔너리입니다.

	Returns:
	str: 딕셔너리를 문자열로 변환한 결과입니다.
	"""
	return pd.DataFrame(my_dict, index=[0]).T.to_string(header=False)



def linux_alert(title, message):
	"""
	Linux에서 경고음을 내고 메시지 박스를 띄우는 함수입니다.

	Args:
	title (str): 메시지 박스의 제목입니다.
	message (str): 메시지 박스에 표시할 메시지입니다.

	Install beep and zenity:
		sudo apt install beep zenity
	"""
	# 경고음 발생
	# subprocess.Popen(['/usr/bin/paplay', sound_file_path])
	# 메시지 박스 띄우기

	# if type of message is dict_module, convert it to string
	if isinstance(message, dict):
		message = convert_dict_to_str(message)

	subprocess.run(['/usr/bin/zenity', '--info', '--title', title, '--text', message])



def linux_notification(title, message):
	"""
	Linux에서 알림을 띄우는 함수입니다.
	다만 notify-send 는 cron 에서 사용하기 어렵습니다.

	Args:
	title (str): 알림의 제목입니다.
	message (str): 알림에 표시할 메시지입니다.

	Install notify-send:
		sudo apt install libnotify-bin
	"""

	# if type of message is dict_module, convert it to string
	if isinstance(message, dict):
		message = convert_dict_to_str(message)

	subprocess.run(['/usr/bin/notify-send', title, message])