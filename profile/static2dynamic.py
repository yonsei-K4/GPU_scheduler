import torch
import torchvision.models as models

# Load a pre-trained ResNet-50
model_dict = {"resnet50": models.resnet50(pretrained=True), 
              "alexnet": models.alexnet(pretrained=True), 
              "vgg19": models.vgg19(pretrained=True), 
              "resnet18": models.resnet18(pretrained=True),
             }

for model_name, model in model_dict.items():
    model.eval()

    # Dummy input with batch size 1
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX with dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=16
    )
