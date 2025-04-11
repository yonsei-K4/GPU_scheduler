#!/bin/bash

# 가상환경 python 경로
VENV_PYTHON="/home/donghyun/miniconda3/envs/K4/bin/python"

echo "Running onnx.py with venv python as root..."
sudo "$VENV_PYTHON" run_onnx.py

# 종료 상태 확인
if [ $? -eq 0 ]; then
  echo "onnx.py 실행 완료!"
else
  echo "onnx.py 실행 중 오류 발생!"
fi
