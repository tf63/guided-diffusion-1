import torch

from adm_model import ADMWrapper

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

model = ADMWrapper()
model.prepare()
batch_size = 1
classes = torch.randint(
    low=0,
    high=NUM_CLASSES,
    size=(batch_size,),
    device=dist_util.dev(),
)

print(f"classes: {classes}")

device = dist_util.dev()
x = torch.randn(batch_size, 3, 64, 64).to(device)
t = torch.tensor([1000]).to(device)  # (batchsize, )である必要がある

model_mean = model(x, t, classes)

print(f"model_mean: {model_mean.shape}")
