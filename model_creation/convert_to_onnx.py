import torch 
from torchvision.models import resnet18
import onnxruntime as ort
import numpy as np

model_weight_path = "model_weights.pth"

# Load PyTorch model
model = resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Get PyTorch output on dummy input
with torch.no_grad():
    pytorch_out = model(dummy_input)
    print("PyTorch output on dummy:", pytorch_out.numpy())

# Export using EXPLICIT legacy exporter
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        dynamo=False  # FORCE legacy exporter
    )

print("Export complete!")

# Verify ONNX export
ort_session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
onnx_out = ort_session.run(None, {"input": dummy_input.numpy()})
print("ONNX output on dummy:", onnx_out[0])

print("\nOutputs match:", np.allclose(pytorch_out.numpy(), onnx_out[0], atol=1e-4))