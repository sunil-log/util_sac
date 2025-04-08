#!/bin/bash


# env variable
home=$(pwd)
export PYTHONPATH="$home/src:$PYTHONPATH"


# 사용할 Python 스크립트 경로
py_script="./src/util_sac/sys/zips/backup_pwd.py"

# 후보로 삼을 Python Interpreter 경로 목록
py_interpreters=(
    "/home/sac/miniconda3/envs/pandas/bin/python"
    "/home/sac/anaconda3/bin/python"
)

# 첫 번째로 실행 파일(-x)로 확인되는 Interpreter를 찾아서 사용
for py in "${py_interpreters[@]}"; do
    if [ -x "$py" ]; then
        py_interpreter="$py"
        break
    fi
done

# Interpreter와 Script 실행
if [ -n "$py_interpreter" ]; then
    "$py_interpreter" "$py_script"
else
    echo "사용 가능한 Python Interpreter를 찾을 수 없습니다."
fi

