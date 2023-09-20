import argparse

from tqdm import tqdm
import torch

from tests.load_model_utils import create_model, get_named_beta_schedule

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
)
from guided_diffusion.sample_utils import save_images

if __name__ == "__main__":
    with torch.no_grad():
        print(
            "Load Model Test ====================================================================="
        )

        device = dist_util.dev()
        print(f"device: {device}")

        # 64x64 model
        model = create_model(
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

        print("create model")

        model_path = "models/64x64_diffusion.pt"
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))

        print("load model")

        model.to(device)
        model.convert_to_fp16()  # if
        model.eval()

        batch_size = 1
        img_size = 64
        timesteps = 1000
        schedule = "cosine"
        classes = torch.randint(
            low=0,
            high=NUM_CLASSES,
            size=(batch_size,),
            device=device,
        )
        model_kwargs = {"y": classes}

        # step test...
        print("step test -------------------------------------------------------------")
        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        t = torch.arange(0, timesteps + 1).to(device)
        t = t.unsqueeze(1).repeat(1, batch_size)  # (batchsize, )である必要がある
        print(f"t: {t}")
        output = model(x, t[timesteps], **model_kwargs)
        output_mean, output_var_values = torch.split(output, 3, dim=1)

        print(f"output: {output.shape}")
        print(f"output_mean: {output_mean.shape}")

        # sampling test
        print(
            "sampling test -------------------------------------------------------------"
        )
        betas = get_named_beta_schedule(schedule, timesteps).to(device)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)

        x = torch.randn(batch_size, 3, img_size, img_size).to(device)
        # t = torch.arange(0, timesteps + 1).to(device)  # (batchsize, )である必要がある
        # t = t.unsqueeze(1).repeat(1, batch_size)

        for i in tqdm(reversed(range(1, timesteps + 1))):
            t = (torch.ones(batch_size) * i).long().to(device)
            output = model(x, t, **model_kwargs)
            output_mean, output_var_values = torch.split(output, 3, dim=1)

            alpha = alphas[i - 1][None, None, None]
            alpha_hat = alpha_hats[i - 1][None, None, None]
            beta = betas[i - 1][None, None, None]

            predicted_noise = (torch.sqrt(1 - alpha_hat) / beta) * (
                x - output_mean * torch.sqrt(alpha)
            )
            noise = torch.randn_like(x)

            x = output_mean + torch.sqrt(beta) * noise
            # x = (
            #     1
            #     / torch.sqrt(alpha)
            #     * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
            #     + torch.sqrt(beta) * noise
            # )

        print(f"sampling complete x: {x.shape}")
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        save_images(x, "a.png")

    # model_epsilon = (sqrt_one_minus_alphabar / beta) * (x - model_mean * sqrt_alpha)
    # sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

    # samples = sample_fn(
    #     model=model,
    #     shape=(args.batch_size, 3, args.image_size, args.image_size),
    #     clip_denoised=args.clip_denoised,
    #     model_kwargs=model_kwargs,
    #     device=dist_util.dev(),
    #     progress=True,
    # )
