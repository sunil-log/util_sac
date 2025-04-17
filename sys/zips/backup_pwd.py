

from datetime import datetime

from util_sac.sys.files.dir_manager import create_dir
from util_sac.sys.zips.zip_df import backup_keywords


def get_comment():
	"""
	0. take a comment from the user
	"""
	comment = input("comment: ")
	print(f"> comment: {comment}")

	# exit if no comment
	if comment == "":
		print(f"> no comment - exit")
		return None

	return comment


def main():

	"""
	1. find files having some keywords in the files name
	2. exclude files having some keywords in the files name
	3. zip them and copy it to ./backup
	"""

	dir_backup = './backup'
	create_dir(dir_backup)


	"""
	0. take a comment from the user
	"""
	zip_comnent = get_comment()

	# key_in and key_out
	key_in = [".py", ".m", ".sh", ".txt", "Dockerfile", '.yaml']
	key_out = [".pyc", ".png", ".npy", "__pycache__", '.npz', '.pkl', '.zip', '.mat']

	# zip all
	now_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	fn_zip = f"{dir_backup}/{now_time}, {zip_comnent}.zip"
	backup_keywords(fn_zip, key_in, key_out, src_loc=".")



if __name__ == '__main__':
	main()
