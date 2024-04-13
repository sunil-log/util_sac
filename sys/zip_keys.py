
"""
zipfile is a standard library for creating and extracting zip archives.
	https://docs.python.org/3/library/zipfile.html
"""
import zipfile
import os


from util_sac.sys.check_dir import FileSystemManager, RenewDirectoryCommand
fs_manager = FileSystemManager()
renew_command = RenewDirectoryCommand(fs_manager)


from glob import glob
import subprocess
import pandas as pd

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
	renew_command.execute(dst)
	for ifn in fns:
		print(f"> unzipping {ifn}")
		subprocess.call(['unzip', ifn, "-d", f"./{dst}/"])




def backup_keywords(fn_zip, key_must, key_in, key_out, src_loc="."):

	"""
	make zip file of given list of files
	"""

	"""
	1. get list of files to zip
		find all files having extension 'py' or 'sh' or 'txt' from the all subdirs of the current dir
	"""

	"""
	absolute path of src_loc
		src_loc = ".." 인 경우가 있음. 이러면 문제가 발생할 수 있어서, 절대경로로 바꿔줌
	"""
	src_loc = os.path.abspath(src_loc)

	# find all files
	fns = glob(f'{src_loc}/**/*.*', recursive=True)
	print(f"found {len(fns)} files")

	# make pandas dataframe
	df = pd.DataFrame(fns, columns=['fn'])


	"""
	2. exclude some files	
	"""
	# take only include "util_sac" and "sys"
	for ikey in key_must:
		df = df[df['fn'].str.contains(ikey)]

	# take only include "py" or "sh" or "txt"
	df = df[df['fn'].str.contains('|'.join(key_in))]

	# exclude "pyc" or "res"
	df = df[~df['fn'].str.contains('|'.join(key_out))]

	# print message for reduced list of files
	print(f"reduced list of files: {df.shape[0]}")
	print(df)


	"""
	3. make zip file of the list of files using zipfile
	"""


	# make zip file
	with zipfile.ZipFile(fn_zip, 'w') as myzip:
		for ifn in df['fn']:
			myzip.write(ifn)

	# print size of the zip file in MB
	print(f"\n> size of the zip file: {os.path.getsize(fn_zip)/1e6:.2f} MB")



def main():
	pass



if __name__ == '__main__':
	main()
