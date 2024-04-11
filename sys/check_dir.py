

import shutil
import os
from pathlib import Path


def check_exists(myPath):

	"""check if the path exists"""

	if not Path(myPath).exists():
		# print(f"{myPath} does not exist")
		return False
	else:
		# print(f"> {myPath} exists")
		return True



def remove_dir(mydir):

	"""remove dir only if it exists"""

	# check if the dir exists
	if check_exists(mydir):
		# remove dir
		print(f"> removing dir: {mydir}")
		shutil.rmtree(mydir)
	else:
		print(f"> removing error: {mydir} does not exist")
		return False



def renew_dir(mydir):

	"""renew dir"""

	# remove dir if it exists
	remove_dir(mydir)

	# the dir does not exist (if or if not existed)

	# create dir
	create_dir(mydir)

	
	

def create_dir(mydir):

	"""create dir"""

	# check if the dir exists
	if check_exists(mydir):
		# do nothing
		print(f"> {mydir} already exists - do nothing")
	else:
		# create dir
		print(f"> creating dir: {mydir}")
		os.mkdir(mydir)


def main():
	fn = './fiifi'
	renew_dir(fn)


if __name__ == '__main__':
	main()

