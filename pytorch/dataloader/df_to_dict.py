

import pandas as pd
import numpy as np

def columns_to_numpy_dict(df: pd.DataFrame) -> dict:
    """
    주어진 df 의 각 컬럼명과 해당 컬럼의 numpy array 를 짝지어
    dict 형태로 반환한다.
    """
    return {col: df[col].to_numpy() for col in df.columns}

# 예시 사용
# df1 은 중복된 ID 에 대해 drop_duplicates 를 수행한 상태라고 가정한다.
# 아래처럼 함수 사용 가능
# result_dict = columns_to_numpy_dict(df1)
# print(result_dict)
