import onnxruntime as ort
import torch

save_onnx = "final_weights/mini_vit.onnx"
input_sample = torch.randn((1, 3, 224, 224))

ort_session = ort.InferenceSession(save_onnx)

input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: input_sample.numpy()}

ort_outs = ort_session.run(None, ort_inputs)

tensor_outputs = torch.tensor(ort_outs)[:, :,-1,:].squeeze()
print(tensor_outputs.shape)
print(tensor_outputs)