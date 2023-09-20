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
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from natsort import natsorted

from guided_diffusion import dist_util, logger
from guided_diffusion.sample_utils import save_images, make_gif_from_tensor, unnorm
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def batch_iterator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def main():
    # settings
    args = create_argparser().parse_args()
    exp_name = args.exp_name
    i_run = args.i_run
    batch_size = args.batch_size
    device = dist_util.dev()
    out_dir = os.path.join("results", "trans", f"{exp_name}_run{i_run}")
    os.makedirs(out_dir, exist_ok=True)

    # setup
    dist_util.setup_dist()
    th.manual_seed(i_run)
    logger.configure(dir=os.path.join(args.log_dir, f"{exp_name}_run{i_run}"))

    logger.log("== resampling =========================================")
    logger.log(f"start {exp_name} run {i_run}")
    logger.log(
        f"dir: {args.img_dir} batch_size: {batch_size} class_cond: {args.class_cond} classifier: True"
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
    # load image
    # -- test --------------------------------------------------------
    # input = th.randn((3, 64, 64)).clamp(-1, 1).to(device)
    # ----------------------------------------------------------------

    # set t
    t_max = args.t_max
    t_batch = th.tensor([t_max]).to(device)

    img_list = []

    for filename in natsorted(os.listdir(args.img_dir)):
        if filename.endswith(".png") and not filename.endswith("_list.png"):
            file_path = os.path.join(args.img_dir, filename)
            img_list.append(file_path)

    all_images = []
    all_labels = []

    for i_batch, img_path_batch in enumerate(batch_iterator(img_list, batch_size)):
        # for k, img_path in enumerate(img_list):
        inputs = []
        for img_path in img_path_batch:
            img = Image.open(img_path)
            img = to_tensor(img)
            inputs.append(img)

        inputs = th.stack(inputs, dim=0).to(device)
        print(f"batch{i_batch} load image: {inputs.shape}")
        inputs = (inputs - 0.5) * 2

        # get x_t
        x_max = diffusion.q_sample(x_start=inputs, t=t_batch)

        inputs = unnorm(inputs)
        save_images(
            inputs,
            os.path.join(out_dir, f"{args.exp_name}_batch{i_batch}_in.png"),
            padding=1,
        )

        # ================================================================

        # resampling
        logger.log(f"resampling... t = {t_max}")

        # settings
        model_kwargs = {}
        classes = th.tensor([int(args.label)] * args.batch_size).to(device)
        model_kwargs["y"] = classes

        # if args.class_cond:
        #     if args.label is not None:
        #         classes = th.tensor([int(args.label)]).to(device) * args.batch_size
        #     else:
        #         classes = th.randint(
        #             low=0,
        #             high=NUM_CLASSES,
        #             size=(args.batch_size,),
        #             device=dist_util.dev(),
        #         )
        #     model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # ここでx_tを指定する
        t = t_max
        logger.log(f"respace t = {t_max} -> {t}")
        samples = sample_fn(
            model=model,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
            noise=x_max,
            t_max=t,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            progress=True,
        )

        # to img
        samples = unnorm(samples)
        sample = samples[-1]

        for i, img in enumerate(sample):
            idf = i_batch * batch_size + i
            save_images(
                img,
                os.path.join(out_dir, f"{args.exp_name}_{idf}.png"),
                padding=0,
            )

        save_images(
            sample,
            os.path.join(out_dir, f"{args.exp_name}_batch{i_batch}_out.png"),
            padding=1,
        )
        print(f"save imgs to {out_dir}")

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

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        exp_name="test",
        log_dir="logs",
        t_max=1000,
        img_dir="",
        i_run=0,
        make_gif=False,
        save_steps=10,
        label=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
