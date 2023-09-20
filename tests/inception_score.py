import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = sorted(os.listdir(directory))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.image_paths[index])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


# Inception Scoreの計算
def compute_inception_score(generated_loader, inception_model, device):
    scores = []
    for batch in tqdm(generated_loader):
        batch = batch.to(device)
        batch = preprocess(batch)
        with torch.no_grad():
            # Inception Scoreの計算
            logits = inception_model(batch)
            probabilities = F.softmax(logits, dim=1)
            kl_divergence = torch.mean(
                torch.sum(
                    probabilities
                    * (
                        torch.log(probabilities)
                        - torch.log(torch.mean(probabilities, dim=0, keepdim=True))
                    ),
                    dim=1,
                )
            )
            score = torch.exp(kl_divergence)
            scores.append(score)

    scores = torch.stack(scores, dim=0)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir", type=str, required=True, help="img directory for evaluation"
    )
    args = parser.parse_args()

    # 生成された画像の確率分布を評価するためのデータセット
    # img_dir = "results/trans/trans64_dir_404to817_t500_run0"
    img_dir = args.img_dir
    generated_dataset = ImageDataset(directory=img_dir, transform=transforms.ToTensor())

    # Inceptionモデルの読み込みと前処理
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model.to(device)

    # データローダーの設定
    batch_size = 64  # 適宜調整してください
    num_workers = 1  # 適宜調整してください
    generated_loader = DataLoader(
        generated_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 画像の前処理
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            # transforms.ToTensor(),
        ]
    )

    # Inception Scoreの計算結果の取得
    inception_score, inception_score_std = compute_inception_score(
        generated_loader, inception_model, device
    )
    print(
        "Inception Score: {:.2f} +/- {:.2f}".format(
            inception_score, inception_score_std
        )
    )
