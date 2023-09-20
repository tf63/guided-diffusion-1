"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms.functional import to_tensor

from guided_diffusion import dist_util, logger
from guided_diffusion.sample_utils import save_images, make_gif_from_tensor, unnorm
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    # settings
    args = create_argparser().parse_args()
    exp_name = args.exp_name
    i_run = args.i_run
    batch_size = args.batch_size
    num_samples = args.num_samples
    device = dist_util.dev()
    out_dir = os.path.join("results", "resample", f"{exp_name}_{i_run}")
    os.makedirs(out_dir, exist_ok=True)

    # setup
    dist_util.setup_dist()
    th.manual_seed(i_run)
    logger.configure(dir=os.path.join(args.log_dir, f"{exp_name}_{i_run}"))

    logger.log("== resampling =========================================")
    logger.log(f"start {exp_name} run {i_run}")
    logger.log(
        f"samples: {num_samples} batch_size: {batch_size} class_cond: {args.class_cond}"
    )
    logger.log("================================================================")

    # create model
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # ================================================================
    # load image
    logger.log(f"open image... path: {args.img_path}")
    # -- test --------------------------------------------------------
    # input = th.randn((3, 64, 64)).clamp(-1, 1).to(device)
    # ----------------------------------------------------------------
    # set t
    t_max = args.t_max
    t_batch = th.tensor([t_max]).to(device)

    input = Image.open(args.img_path)
    input = to_tensor(input)[None].to(device)
    input = (input - 0.5) * 2
    print(f"load image: {input.shape}")

    # get x_t
    x_max = diffusion.q_sample(x_start=input, t=t_batch)
    # logger.log(x_max)

    input = unnorm(input)
    img_noisy = unnorm(x_max)
    save_images(
        img_noisy,
        os.path.join(out_dir, f"{args.exp_name}_noisy.png"),
        padding=0,
    )

    # ================================================================

    # resampling
    logger.log(f"resampling... t = {t_max}")
    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:
        # settings
        model_kwargs = {}
        if args.class_cond:
            if args.label is not None:
                classes = th.tensor([int(args.label)]).to(device) * args.batch_size
            else:
                classes = th.randint(
                    low=0,
                    high=NUM_CLASSES,
                    size=(args.batch_size,),
                    device=dist_util.dev(),
                )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # ここでx_tを指定する
        # t = t_max * int(args.timestep_respacing) / int(args.diffusion_steps)
        # t = int(t)
        # t = 200
        t = t_max
        logger.log(f"respace t = {t_max} -> {t}")
        samples = sample_fn(
            model=model,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
            noise=x_max,
            t_max=t,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )

        # to img
        samples = unnorm(samples)
        sample = samples[-1]

        # gifにする
        logger.log(f"sample shape: {sample.shape}")
        logger.log(f"samples shape: {samples.shape}")
        for i, sample_list in enumerate(samples.split(1, dim=1)):
            sample_list = th.squeeze(sample_list)
            logger.log(f"sample_list shape: {sample_list[-1][None].shape}")
            sample_list_slice = th.unsqueeze(sample_list[:: args.save_steps], 1)
            sample_list = th.cat([*sample_list_slice, sample_list[-1][None]])
            logger.log(f"sample_list shape: {sample_list.shape}")
            if args.make_gif:
                make_gif_from_tensor(sample_list, out_dir, f"{args.exp_name}_{i}")

            save_images(
                sample_list,
                os.path.join(out_dir, f"{args.exp_name}_{i}_list.png"),
                padding=1,
            )

        # まとめる?
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    imgs = th.cat(all_images, dim=0)
    imgs = imgs[:num_samples]
    logger.log(f"all_images: {imgs.shape}")

    if args.class_cond:
        labels = th.cat(all_labels, dim=0)
        labels = labels[:num_samples]
        logger.log(f"all_labels: {labels.shape}")

    logger.log(f"saving to {out_dir}")
    for i, img in enumerate(imgs):
        img = img[None]
        save_images(
            img,
            os.path.join(out_dir, f"{args.exp_name}_{i}.png"),
            padding=0,
        )
        img_cmp = th.cat([input, img])
        save_images(
            img_cmp,
            os.path.join(out_dir, f"{args.exp_name}_{i}_cmp.png"),
            padding=1,
        )

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        exp_name="test",
        log_dir="logs",
        t_max=1000,
        img_path="",
        i_run=0,
        make_gif=False,
        save_steps=10,
        label=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
