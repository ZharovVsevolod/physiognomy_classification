from models.shell import Model_Lightning_Shell
import torch

model = Model_Lightning_Shell.load_from_checkpoint("outputs/2024-05-04/21-25-13/weights/epoch_epoch=14-val_loss=0.0173.ckpt")

save_onnx = "final_weights/mini_vit.onnx"
input_sample = torch.randn((1, 3, 224, 224))

model.to_onnx(save_onnx, input_sample, export_params=True)