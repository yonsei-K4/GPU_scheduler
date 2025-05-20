import torch
import timm
import os
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, ResNet101_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights,
    SqueezeNet1_0_Weights, ShuffleNet_V2_X1_0_Weights,
    MobileNet_V2_Weights, DenseNet121_Weights, DenseNet201_Weights,
    Inception_V3_Weights, EfficientNet_B7_Weights,
)

model_dir = "onnx_models"
os.makedirs(model_dir, exist_ok=True)

# torchvision 모델 정의 (모델 함수, weight enum, 입력 사이즈)
torchvision_models = {
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT, 224),
    "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT, 224),
    "resnext50_32x4d": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.DEFAULT, 224),
    "resnext101_32x8d": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.DEFAULT, 224),
    "squeezenet1_0": (models.squeezenet1_0, SqueezeNet1_0_Weights.DEFAULT, 224),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.DEFAULT, 224),
    "mobilenet_v2": (models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT, 224),
    "densenet121": (models.densenet121, DenseNet121_Weights.DEFAULT, 224),
    "densenet201": (models.densenet201, DenseNet201_Weights.DEFAULT, 224),
    "inception_v3": (models.inception_v3, Inception_V3_Weights.DEFAULT, 299),
    "efficientnet_b7": (models.efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, 600),
}

# timm 모델 정의 (모델 이름, 입력 사이즈)
timm_models = {
    "inception_v4": 299,
    "inception_resnet_v2": 299,
}

# torchvision 변환
for name, (model_fn, weights_enum, img_size) in torchvision_models.items():
    try:
        print(f"🔄 torchvision - {name} 변환 중...")
        model = model_fn(weights=weights_enum)
        model.eval()

        dummy_input = torch.randn(1, 3, img_size, img_size)
        path = f"{model_dir}/{name}.onnx"

        torch.onnx.export(
            model, dummy_input, path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,
            do_constant_folding=True
        )
        print(f"✅ {name} → {path}")
    except Exception as e:
        print(f"❌ {name} 변환 실패: {e}")

# timm 변환
for name, img_size in timm_models.items():
    try:
        print(f"🔄 timm - {name} 변환 중...")
        model = timm.create_model(name, pretrained=True)
        model.eval()

        dummy_input = torch.randn(1, 3, img_size, img_size)
        path = f"{model_dir}/{name}.onnx"

        torch.onnx.export(
            model, dummy_input, path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,
            do_constant_folding=True
        )
        print(f"✅ {name} → {path}")
    except Exception as e:
        print(f"❌ {name} 변환 실패: {e}")
