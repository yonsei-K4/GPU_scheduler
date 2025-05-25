#!/usr/bin/env python3
# single_inference.py
import argparse
import time
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from datasets import load_dataset
import pathlib
import sys
# --------------------------------------------------
# 0. 환경 설정
# --------------------------------------------------
# print(">>> import path  :", pathlib.Path(ort.__file__).parent)
# print(">>> package ver :", ort.__version__)
# print(">>> sys.path[0] :", sys.path[0])   # 현재 작업 디렉터리

# --------------------------------------------------
# 1. CLI
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Single ONNX inference")
parser.add_argument("--model", default="onnx_models/resnet152-v1-7.onnx",
                    help="ONNX 모델 경로")
args = parser.parse_args()

# --------------------------------------------------
# 2. 이미지 로드 & 전처리
# --------------------------------------------------
dataset = load_dataset("huggingface/cats-image", split="test")
image = dataset[0]["image"]           # PIL.Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),            # (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_np = preprocess(image).unsqueeze(0).numpy()  # (1, 3, 224, 224)

# --------------------------------------------------
# 3. ONNX Runtime 세션
# --------------------------------------------------
so = ort.SessionOptions()
# so.enable_profiling = True            # ort_profile_<pid>_<time>.json 생성
session = ort.InferenceSession(
    args.model, sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# 안전망
assert img_np.shape == (1, 3, 224, 224) and img_np.dtype == np.float32

# --------------------------------------------------
# 4. 단 1회 추론
# --------------------------------------------------
start = time.time()
outputs = session.run(None, {input_name: img_np})
latency_ms = (time.time() - start) * 1e3

#print(f"✅ inference latency: {latency_ms:,.2f} ms")
#print("output tensor shape:", outputs[0].shape)

# --------------------------------------------------
# 5. ORT 프로파일 파일 저장 위치 안내
# --------------------------------------------------
# profile_path = session.end_profiling()
# print("ONNX Runtime profile saved to:", profile_path)
