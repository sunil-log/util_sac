


def print_partial_markdown(df, n_rows=10, file_name=None):
    """
    DataFrame을 Markdown 형태로 부분 출력하고, file_name이 주어지면 전체 데이터를 파일에 저장합니다.

    Args:
        df (pd.DataFrame): Markdown으로 변환할 DataFrame.
        n_rows (int, optional): DataFrame에서 상단과 하단에 출력할 행의 개수. 기본값은 10입니다.
        file_name (str, optional): 지정할 경우, 전체 DataFrame을 Markdown 형태로 file_name에 저장합니다.
                                  주어지지 않으면 파일로 저장하지 않습니다.

    Returns:
        None
    """
    # DataFrame을 전체 Markdown 문자열로 변환
    markdown_str = df.to_markdown()

    # 화면에 출력할 부분(상·하단 n_rows)
    lines = markdown_str.split('\n')
    if len(lines) < 2 * n_rows:
        partial_str = markdown_str
    else:
        # 상위 n_rows + 헤더 2행 + ... + 하위 n_rows
        selected_lines = lines[:n_rows + 2] + ['...'] + lines[-n_rows:]
        partial_str = '\n'.join(selected_lines)

    # 부분 출력
    print(partial_str)
    print(f"Total number of rows: {len(df)}")

    # file_name이 주어지면 전체 Markdown을 파일에 저장
    if file_name is not None:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(markdown_str)
            f.write(f"\nTotal number of rows: {len(df)}\n")
