#!/usr/bin/env python3
# single_inference.py
import argparse
import time
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from datasets import load_dataset
import pynvml
import threading

# GPU 메모리 사용량을 추적하기 위한 pynvml 초기화
pynvml.nvmlInit()
gpu_id = 0  # CUDA device_id, 0번 GPU

handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
mem_trace = []
trace_flag = True

def trace_memory(interval=0.01):  # 5ms 간격 샘플링
    while trace_flag:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / (1024 * 1024)
        mem_trace.append((time.time(), used_mb))
        time.sleep(interval)

# --------------------------------------------------
# 0. 환경 설정
# --------------------------------------------------
# print(">>> import path  :", pathlib.Path(ort.__file__).parent)
# print(">>> package ver :", ort.__version__)
# print(">>> sys.path[0] :", sys.path[0])   # 현재 작업 디렉터리

# --------------------------------------------------
# 1. CLI
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Single ONNX inference with memory tracing")
parser.add_argument("--model", required=True, type=str,
                    help="ONNX 모델 이름 (예: resnet50)")
parser.add_argument("--batch", type=int, default=1,
                    help="입력 배치 크기 (기본값: 1)")
args = parser.parse_args()

batch_size = args.batch
model_dir = f"/models/{args.model}.onnx"
print(f"✅ model: {model_dir}, batch: {batch_size}")

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
img_np = np.repeat(img_np, repeats=batch_size, axis=0)  # (B, 3, 224, 224)

# --------------------------------------------------
# 3. ONNX Runtime 세션
# --------------------------------------------------
so = ort.SessionOptions()
# so.enable_profiling = True            # ort_profile_<pid>_<time>.json 생성
session = ort.InferenceSession(
    model_dir, sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# 안전망
assert img_np.shape == (batch_size, 3, 224, 224) and img_np.dtype == np.float32

# --------------------------------------------------
# 4. 단 1회 추론
# --------------------------------------------------
trace_thread = threading.Thread(target=trace_memory)
trace_thread.start()

start = time.time()
outputs = session.run(None, {input_name: img_np})
latency_ms = (time.time() - start) * 1e3

trace_flag = False
trace_thread.join()

print(f"✅ inference latency: {latency_ms:,.2f} ms")
print("output tensor shape:", outputs[0].shape)

# 6. 메모리 사용량 기록 출력
start_t = mem_trace[0][0]
for t, mem in mem_trace:
    print(f"{(t - start_t)*1e3:.1f} ms : {mem:.2f} MB")

# --------------------------------------------------
# 5. ORT 프로파일 파일 저장 위치 안내
# --------------------------------------------------
# profile_path = session.end_profiling()
# print("ONNX Runtime profile saved to:", profile_path)
