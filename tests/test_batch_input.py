from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from natsort import natsorted
import os


def batch_iterator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


img_dir = "results/image/tcfg64_label817_run0"
device = "cuda"
batch_size = 16
img_list = []

for filename in natsorted(os.listdir(img_dir)):
    if filename.endswith(".png") and not filename.endswith("_list.png"):
        file_path = os.path.join(img_dir, filename)
        img_list.append(file_path)

for i_batch, img_path_batch in enumerate(batch_iterator(img_list, batch_size)):
    inputs = []
    for img_path in img_path_batch:
        # print(f"input img path: {img_path}")
        img = Image.open(img_path)
        img = to_tensor(img)
        inputs.append(img)

    inputs = torch.stack(inputs, dim=0)
    print(f"batch{i_batch} load image: {inputs.shape}")
