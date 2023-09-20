"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.sample_utils import save_images, make_gif_from_tensor
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
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
    out_dir = os.path.join("results", "classifier", f"{exp_name}_{i_run}")
    os.makedirs(out_dir, exist_ok=True)

    # setup
    dist_util.setup_dist()
    th.manual_seed(i_run)
    logger.configure(dir=os.path.join(args.log_dir, f"{exp_name}_{i_run}"))

    logger.log("== classifier sampling =========================================")
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
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # ================================================================

    # sampling
    logger.log("sampling...")
    all_images = []
    all_labels = []

    while len(all_images) * batch_size < num_samples:
        # settings
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # sampling
        samples = sample_fn(
            model_fn,
            (batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            progress=True,
        )

        # to img
        samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
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
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels for labels in gathered_labels])
        logger.log(f"created {len(all_images) * batch_size} samples")

    imgs = th.cat(all_images, dim=0)
    imgs = imgs[:num_samples]
    labels = th.cat(all_labels, dim=0)
    labels = labels[:num_samples]

    logger.log(f"all_images: {imgs.shape}")
    logger.log(f"all_labels: {labels.shape}")

    logger.log(f"saving to {out_dir}")
    save_images(
        imgs,
        os.path.join(out_dir, f"{args.exp_name}_list.png"),
        padding=1,
    )
    for i, img in enumerate(imgs):
        save_images(
            img,
            os.path.join(out_dir, f"{args.exp_name}_{i}.png"),
            padding=0,
        )

    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: num_samples]
    # label_arr = np.concatenate(all_labels, axis=0)
    # label_arr = label_arr[: num_samples]
    # if dist.get_rank() == 0:
    #     os.makedirs(out_dir, exist_ok=True)
    #     logger.log(f"saving to {out_dir}")
    #     # shape_str = "x".join([str(x) for x in arr.shape])
    #     # np.savez(os.path.join(out_dir, f"samples_{shape_str}.npz"), arr, label_arr)

    #     print(arr.shape)
    #     for i in range(arr.shape[0]):
    #         logger.log(f"save {i}.png")
    #         img = Image.fromarray(arr[i])
    #         img.save(os.path.join(out_dir, f"{args.exp_name}-{i}.png"))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        exp_name="test_sample",
        log_dir="logs",
        i_run=0,
        make_gif=False,
        save_steps=10,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
