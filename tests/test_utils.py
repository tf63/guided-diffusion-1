from guided_diffusion.sample_utils import save_images, make_gif_from_tensor
import os
import torch


def test_save_images():
    img = torch.randn((1, 3, 64, 64))
    img = (img.clamp(-1, 1) + 1) / 2
    img = (img * 255).type(torch.uint8)
    path = "results/test_utils"
    os.makedirs(path, exist_ok=True)
    save_images(img, os.path.join(path, "test.png"), nrow=7, padding=1)


def test_make_gif_from_tensor():
    img = torch.randn((100, 3, 64, 64))
    img = (img.clamp(-1, 1) + 1) / 2
    img = (img * 255).type(torch.uint8)
    path = "results/test_utils"
    name = "testgif"
    os.makedirs(path, exist_ok=True)
    make_gif_from_tensor(img, path, name)


if __name__ == "__main__":
    # test_save_images()
    test_make_gif_from_tensor()
