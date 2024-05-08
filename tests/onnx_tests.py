import onnxruntime as ort
import torch
from skimage import io
import os
from torchvision import transforms

def read_image(img_name, root_dir):
    img_path = root_dir + "/" + img_name
    image = io.imread(img_path) / 255
    image = torch.Tensor(image).float()
    image = image.reshape(3, 224, 224)
    return image

def get_images(default_dir:str|None = None, debug:bool = False):
    if debug:
        return torch.randn((1, 3, 224, 224))

    transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    if default_dir is None:
        root_dir = "dataset/faces"
    else:
        root_dir = default_dir
    
    image_names = os.listdir(root_dir)

    if len(image_names) == 1:
        img_name = image_names[0]
        print(f"Find the image named {img_name}")
        image = read_image(img_name, root_dir)
        image = image.unsqueeze(0) # image.shape = [1, 3, 224, 224]
        image = transform(image)
        names = img_name
    else:
        all_images = []
        names = []
        for img_name in image_names:
            image = read_image(img_name, root_dir)
            image = transform(image)
            all_images.append(image)
            names.append(img_name)
        image = torch.stack(all_images)

    return image, names

def print_ocean(ocean, name):
    print(f"OCEAN for {name}")
    print(f"O - Openness = {ocean[0]}")
    print(f"C - Conscientiousness = {ocean[1]}")
    print(f"E - Extraversion = {ocean[2]}")
    print(f"A - Agreeableness = {ocean[3]}")
    print(f"N - Neuroticism = {ocean[4]}")
    print("-----")

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

save_onnx = "final_weights/mini_vit.onnx"
image, names = get_images()

ort_session = ort.InferenceSession(save_onnx)
input_name = ort_session.get_inputs()[0].name

pc = 0
for img in image:
    img = img.unsqueeze(0)
    ort_inputs = {input_name: img.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    tensor_outputs = torch.tensor(ort_outs)[:, :,-1,:].squeeze()
    print_ocean(tensor_outputs, names[pc])
    pc += 1