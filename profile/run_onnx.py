import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import time
import requests
import argparse

# 0. 명령줄 인자 처리
parser = argparse.ArgumentParser(description="ONNX Inference with interval control")
parser.add_argument('--interval', type=float, default=0.05,
                    help='요청 간격 (초 단위, 기본값: 0.05)')
parser.add_argument('--duration', type=int, default=60,
                    help='총 실행 시간 (초 단위, 기본값: 60)')
args = parser.parse_args()

interval = args.interval
duration = args.duration

# 1. 이미지 데이터셋 로드 (Hugging Face cats 이미지)
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]  # PIL.Image 형식

# 2. 전처리
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(image).unsqueeze(0)  # 배치 차원 추가
img_np = img_tensor.numpy()

# 3. ONNX 모델 로드
session = onnxruntime.InferenceSession(
    "resnext101_32x8d.onnx",
    providers=[
        # "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

# 4. 클래스 레이블 (사용 안 해도 무방)
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = requests.get(labels_url).text.strip().split("\n")

# 5. 반복 수행
start_time = time.time()
inference_count = 0

print(f"🚀 {interval:.2f}초 간격으로 요청 전송 시작 (총 {duration}초 동안)...")

while time.time() - start_time < duration:
    _ = session.run(None, {"input": img_np})
    inference_count += 1
    time.sleep(interval)

print(f"\n✅ 총 요청 수: {inference_count}회 (간격: {interval:.2f}s, 총 {duration}s)")
