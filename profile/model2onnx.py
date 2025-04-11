import torch
from torchvision import models

# 1. 모델 불러오기 (사전 학습된 ResNeXt-101 32x8d)
model = models.resnext101_32x8d(pretrained=True)
model.eval()  # 평가 모드로 설정

# 2. 더미 입력 생성 (배치 크기 1, 채널 3, 이미지 크기 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# # 3. 모델을 GPU로 옮기고 입력도 GPU에 올리기 (선택)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# dummy_input = dummy_input.to(device)

# 4. ONNX로 내보내기
torch.onnx.export(
    model,                                # PyTorch 모델
    dummy_input,                          # 더미 입력
    "resnext101_32x8d.onnx",              # 저장 파일명
    input_names=["input"],                # 입력 이름
    output_names=["output"],              # 출력 이름
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 배치 사이즈 유동성
    opset_version=13,                     # ONNX opset 버전
    do_constant_folding=True              # 최적화 수행
)

print("✅ ResNeXt-101 모델이 ONNX 형식으로 저장되었습니다.")