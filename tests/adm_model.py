import argparse

import torch
import torch.nn as nn

from guided_diffusion.unet import UNetModel
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


class ADMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        model = self.create_model(
            image_size=64,
            num_channels=192,
            num_res_blocks=3,
            num_head_channels=64,
            use_scale_shift_norm=True,
            use_fp16=True,
            use_new_attention_order=True,
            resblock_updown=True,
            learn_sigma=True,
            dropout=0.1,
            class_cond=True,
            use_checkpoint=True,
            attention_resolutions="32,16,8",
        )
        self.model = model
        self.device = dist_util.dev()
        self.model_path = "models/tiny_imagenet_cond_gaussian_k5_s5/model060000.pt"

    def forward(self, x, t, label=None):
        # model_kwargs = {}
        # if label is not None:
        #     model_kwargs = {"y": label}
        y = torch.tensor([label] * x.shape[0]).to(self.device)
        # model_output = self.model(x, t, **model_kwargs)
        model_output = self.model(x, t, y)
        model_mean, model_var_values = torch.split(model_output, 3, dim=1)
        # model_epsilon = (sqrt_one_minus_alphabar / beta) * (x - model_mean * sqrt_alpha)

        return model_mean

    def prepare(self):
        self.model.load_state_dict(
            dist_util.load_state_dict(self.model_path, map_location="cpu")
        )
        self.model.to(self.device)
        self.model.convert_to_fp16()  # if
        self.model.eval()

    def create_model(
        self,
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    ):
        if channel_mult == "":
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return UNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
