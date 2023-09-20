import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3
from torchvision.models.inception import InceptionOutputs
import numpy as np


def sqrtm(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    sqrt_matrix = eigenvectors.dot(np.diag(sqrt_eigenvalues)).dot(
        np.linalg.inv(eigenvectors)
    )
    return sqrt_matrix


def is_complex_tensor(tensor):
    return tensor.dtype in [torch.complex64, torch.complex128]


def calculate_fid(images_real, images_fake, device="cuda"):
    inception_model = (
        inception_v3(pretrained=True, transform_input=False).to(device).eval()
    )

    def get_activations(images, model):
        activations = []
        dataloader = DataLoader(images, batch_size=32, shuffle=False, num_workers=4)
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                output = model(batch)[0].view(batch.shape[0], -1)
            activations.append(output)
        activations = torch.cat(activations, dim=0)
        return activations

    activations_real = get_activations(images_real, inception_model)
    activations_fake = get_activations(images_fake, inception_model)

    mu_real, sigma_real = activations_real.mean(dim=0), torch_cov(
        activations_real, rowvar=False
    )
    mu_fake, sigma_fake = activations_fake.mean(dim=0), torch_cov(
        activations_fake, rowvar=False
    )

    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake)

    if is_complex_tensor(covmean):
        covmean = covmean.real

    fid_score = (diff @ diff) + torch.trace(sigma_real + sigma_fake - 2 * covmean)

    return fid_score.item()


def calculate_is(images, model, device="cuda", num_samples=50000):
    model.eval()

    def kl_div(p, q):
        return torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)

    dataloader = DataLoader(images, batch_size=32, shuffle=False, num_workers=4)
    preds = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
        preds.append(output.softmax(dim=1))
    preds = torch.cat(preds, dim=0)[:num_samples]

    p_y = preds.mean(dim=0)
    kl_divs = kl_div(preds, p_y.unsqueeze(0)).mean()

    return torch.exp(kl_divs).item()


def torch_cov(m, rowvar=False, inplace=False):
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")

    if m.dim() < 2:
        m = m.view(1, -1)

    if not rowvar and m.size(0) != 1:
        m = m.t()

    factor = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    cov = factor * m.matmul(mt).squeeze()

    if inplace:
        return cov

    return cov.clone()


def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                image_files.append(os.path.join(root, file))
    return image_files


# ディレクトリの指定
image_directory = "results/trans/trans64_dir_404to817_t500_run0"

# 画像ファイルの取得
image_files = get_image_files(image_directory)

# 画像データセットの作成
dataset = ImageFolder(root=image_directory, transform=ToTensor())

# 画像データセットから本物画像のみを抽出
images_real = [data[0] for data in dataset]

# 仮想的に生成した画像データを用意（ここでは本物画像と同じ数だけランダムに選択）
images_fake = torch.stack(
    [images_real[i % len(images_real)] for i in range(len(images_real))]
)

# FIDの計算
fid_score = calculate_fid(images_real, images_fake)
print("FID score:", fid_score)

# ISの計算
inception_model = inception_v3(pretrained=True).to("cuda").eval()
is_score = calculate_is(dataset, inception_model)
print("IS score:", is_score)
