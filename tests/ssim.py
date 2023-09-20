import os
import torch
from torchvision import transforms
from natsort import natsorted
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as ssim
import argparse


def get_natsorted_img_list(img_dir):
    img_list = []
    for filename in natsorted(os.listdir(img_dir)):
        if (
            filename.endswith(".png")
            and not filename.endswith("_list.png")
            and not filename.endswith("_in.png")
            and not filename.endswith("_out.png")
        ):
            file_path = os.path.join(img_dir, filename)
            img_list.append(file_path)

    return img_list


parser = argparse.ArgumentParser()

parser.add_argument(
    "--img_dir_base", type=str, required=True, help="img directory for evaluation 1"
)
parser.add_argument(
    "--img_dir_trans",
    type=str,
    required=True,
    help="img directory for evaluation 2",
)
args = parser.parse_args()

img_dir_base = args.img_dir_base
img_list_base = get_natsorted_img_list(img_dir_base)
img_dir_trans = args.img_dir_trans
img_list_trans = get_natsorted_img_list(img_dir_trans)
assert len(img_list_base) == len(img_list_trans)

num_images = len(img_list_base)


# 画像の読み込みに使用する関数を定義します
def load_image(file_path):
    with open(file_path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# 画像をテンソルに変換するためのTransformを定義します
transform = transforms.ToTensor()


# SSIMを計算する関数を定義します
# def calculate_ssim(image1, image2):
# # 画像をテンソルに変換
# tensor_image1 = transform(image1).unsqueeze(0).float()  # バッチ次元を追加
# tensor_image2 = transform(image2).unsqueeze(0).float()

# # テンソルをGPUに転送（必要に応じて）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor_image1 = tensor_image1.to(device)
# tensor_image2 = tensor_image2.to(device)
# print(tensor_image1.max(), tensor_image2.min)
# SSIMを計算
# ssim_value = ssim(tensor_image1, tensor_image2, data_range=1.0, multichannel=True)

# return ssim_value


# 2つのディレクトリから画像を読み込み、SSIMを計算
ssim_sum = 0.0

for i in range(num_images):
    # 画像の読み込み
    # image1 = load_image(img_list_base[i])
    # image2 = load_image(img_list_trans[i])
    image1 = io.imread(img_list_base[i])
    image2 = io.imread(img_list_trans[i])
    # SSIMを計算
    # ssim_value = calculate_ssim(image1, image2)

    ssim_value = ssim(image1, image2, win_size=3)

    ssim_sum += ssim_value
    print(f"{i} SSIM: {ssim_value}")
# 平均SSIMを計算
mean_ssim = ssim_sum / num_images
print("平均SSIM:", mean_ssim)
