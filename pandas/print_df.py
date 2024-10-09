

def print_partial_markdown(df, n_rows=10):
	"""
	Prints the top and bottom n_rows of the dataframe in Markdown format with ellipses in between.

	Args:
	df (pd.DataFrame): The dataframe to be printed.
	n_rows (int): Number of rows to display from the top and bottom of the dataframe.

	Returns:
	None
	"""
	# 데이터프레임을 Markdown 형식으로 변환
	markdown_str = df.to_markdown()

	if n_rows is None:
		print(markdown_str)
		print(f"Total number of rows: {len(df)}")
		return

	# 개행 문자로 분리하여 행 리스트로 변환
	lines = markdown_str.split('\n')

	if len(lines) < 2*n_rows:
		print(markdown_str)
		print(f"Total number of rows: {len(df)}")
		return

	else:
		# 상위 n_rows + 헤더 행 + 하위 n_rows 선택
		# 헤더와 구분선(---|---|--- 형태)은 항상 첫 두 행에 위치
		selected_lines = lines[:n_rows + 2] + ['...'] + lines[-n_rows:]

		# 선택된 행들을 다시 하나의 문자열로 합치기
		result_str = '\n'.join(selected_lines)

		print(result_str)
		print(f"Total number of rows: {len(df)}")
		return
