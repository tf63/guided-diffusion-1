import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Swap amplitude spectrum between two images
def amp_swap(images1, images2):
    fft1 = torch.fft.rfftn(images1, dim=(-1, -2))
    fft2 = torch.fft.rfftn(images2, dim=(-1, -2))
    amp1 = torch.abs(fft1)
    amp2 = torch.abs(fft2)
    phase1 = torch.angle(fft1)
    phase2 = torch.angle(fft2)
    amp2_phase1 = torch.fft.irfftn(amp2 * torch.exp(1j * phase1), dim=(-1, -2))
    amp1_phase2 = torch.fft.irfftn(amp1 * torch.exp(1j * phase2), dim=(-1, -2))
    return amp2_phase1, amp1_phase2


# Split into amp and phase
def split_amp_phase(image):
    if torch.is_tensor(image) is False:
        image = torch.tensor(image)
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    amp_list = []
    phase_list = []
    for i, channel in enumerate(image):
        fft = torch.fft.fft2(channel)
        fftshift = torch.fft.fftshift(fft)
        amp = torch.abs(fftshift)
        phase = torch.angle(fftshift)
        amp_list.append(amp)
        phase_list.append(phase)
    amp_list = torch.stack(amp_list)
    phase_list = torch.stack(phase_list)
    return amp_list, phase_list


if __name__ == "__main__":
    img = Image.open(
        "/home/tfukuda/project/guided-diffusion/docs/img/test_pil64/test_pil64-4.png"
    )

    img_gray = img.convert("L")
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float()

    fft_output = torch.fft.fft2(img_tensor, dim=(-2, -1))

    fft_output_real = fft_output.real
    fft_output_imag = fft_output.imag

    print(
        torch.log(1 + torch.abs(fft_output)).max(),
        torch.log(1 + torch.abs(fft_output)).min(),
    )
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_np, cmap="gray")
    plt.title("Input Image")
    plt.subplot(122)
    plt.imshow(torch.log(1 + torch.abs(fft_output)), cmap="gray")  # ログスケールで表示
    plt.title("FFT Result")
    plt.show()
