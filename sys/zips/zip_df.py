
"""
zipfile is a standard library for creating and extracting zip archives.
	https://docs.python.org/3/library/zipfile.html
"""
import os
import subprocess
import zipfile
from glob import glob

import pandas as pd

from util_sac.sys.dir_manager import renew_dir
from util_sac.sys.search_files import search_items_df

# change pandas maximum width option
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)



def unzip_all(src, dst):

	# find list of notion exports
	fns = glob(f'./{src}/*.zip')
	if len(fns) < 1:
		print("no zip file placed in notion")

	# unzip all
	renew_dir(dst)
	for ifn in fns:
		print(f"> unzipping {ifn}")
		subprocess.call(['unzip', ifn, "-d", f"./{dst}/"])



def zip_files_df(df, fn_zip):

	"""
	                             File Path             Parent                        Stem
	1                        docker_run.sh                  .                  docker_run
	2                 docker_entrypoint.sh                  .           docker_entrypoint
	5                           bus_all.py                  .                     bus_all
	6            src/99_batch_size_test.py                src          99_batch_size_test
	7       src/03_collect_channel_data.py                src     03_collect_channel_data
	"""

	# apply absolute path
	df['Path'] = df['Path'].apply(lambda x: os.path.abspath(x))

	# remove common path
	common_path = os.path.commonpath(df['Path'].tolist())

	# ZIP 파일 만들기
	with zipfile.ZipFile(fn_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
		for file_path in df['Path']:
			# 공통 경로 제거하기
			arcname = os.path.relpath(file_path, common_path)
			print(f"adding {arcname}")
			zf.write(file_path, arcname=arcname)

	# print size of the zip file in MB
	print(f"\n> size of the zip file: {os.path.getsize(fn_zip)/1e6:.2f} MB")



def backup_keywords(fn_zip, key_in, key_out, src_loc="."):

	"""
	1. find files having some keywords in the file name
	2. exclude files having some keywords in the file name
	3. zip them and copy it to ./backup
	"""

	# find files having some keywords in the file name
	pattern = re
	df = search_items_df(src_loc, r"*")
	print(df)
	exit()
	"""
		                                               File Path                        Parent                               Stem
	0                                                 Dockerfile                             .                         Dockerfile
	1                                              docker_run.sh                             .                         docker_run
	2                                       docker_entrypoint.sh                             .                  docker_entrypoint
	
	# key_in and key_out
	key_in = [".py", ".m", ".sh", ".txt", "Dockerfile"]
	key_out = [".pyc", ".png", ".npy", "__pycache__", '.npz', '.pkl', '.zip', '.mat']
	"""

	# key_in and key_out
	df = df[df['Path'].apply(lambda x: any([k in str(x) for k in key_in]))]
	df = df[~df['Path'].apply(lambda x: any([k in str(x) for k in key_out]))]

	# zip all
	zip_files_df(df, fn_zip)


def main():
	pass


if __name__ == '__main__':
	main()
