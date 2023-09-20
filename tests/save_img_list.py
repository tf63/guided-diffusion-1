import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image
from guided_diffusion.sample_utils import save_images, unnorm

# def save_images(images, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)
# img_indexs = [191, 15, 44, 96, 169, 171]
# img_indexs = [5, 0, 1, 2, 3, 4]
img_paths = [
    "results/image/tcfg64_label404_run0/tcfg64_label404_0.png",
    "results/tcfg/tcfg64_i0_404to817_t500_cfg1_0/tcfg64_i0_404to817_t500_cfg1_0.png",
    "results/tcfg/tcfg64_i0_404to817_t500_0/tcfg64_i0_404to817_t500_0.png",
    "results/tcfg/tcfg64_i0_404to817_t500_cfg5_0/tcfg64_i0_404to817_t500_cfg5_0.png",
    "results/tcfg/tcfg64_i0_404to817_t500_cfg15_0/tcfg64_i0_404to817_t500_cfg15_0.png",
    "results/tcfg/tcfg64_i0_404to817_t500_cfg50_0/tcfg64_i0_404to817_t500_cfg50_0.png",
]

imgs = []
for img_path in img_paths:
    img = Image.open(img_path)
    img = to_tensor(img)
    imgs.append(img)

# for img_index in img_indexs:
# img_path = f"results/trans/cg64_dir_404to817_t500_run0/cg64_dir_404to817_t500_{img_index}.png"
#     img_path = f"results/image/tcfg64_label404_run0/tcfg64_label404_{img_index}.png"
# img_path = f"results/image/tcfg64_label817_run0/tcfg64_label817_{img_index}.png"
# img_path = f"results/trans/191_trans_run0/191_trans_{img_index}.png"
# img = Image.open(img_path)
# img = to_tensor(img)
# imgs.append(img)

imgs = torch.stack(imgs, dim=0)
# imgs = torch.randn(6, 3, 64, 64)
# imgs = unnorm(imgs)
imgs = (imgs * 255).to(torch.uint8)
save_images(imgs, "docs/img/presentation/404to817_scale_r.png", padding=1, nrow=6)
