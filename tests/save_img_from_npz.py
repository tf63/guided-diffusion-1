import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2))
    plt.savefig('a.png')


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    print(grid.shape)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    print(ndarr.shape)
    im = Image.fromarray(ndarr)
    im.save(path)

imgs = np.load('/tmp/openai-2023-03-30-23-31-52-723687/samples_4x256x256x3.npz')
print(imgs.files)
x = imgs['arr_0']
y = imgs['arr_1']


for i in range(4):
    plt.axis('off')
    plt.imshow(x[i])
    plt.savefig(f'a256-{i}.png')
