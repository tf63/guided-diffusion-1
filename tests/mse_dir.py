import os
import argparse
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


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


def calculate_mse(image1, image2):
    # 画像の読み込みと前処理
    transform = transforms.ToTensor()
    image1 = transform(image1)
    image2 = transform(image2)

    # MSEを計算
    mse = F.mse_loss(image1, image2)

    return mse.item()


if __name__ == "__main__":
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

    N = len(img_list_base)

    count = 0
    for i in range(N):
        if img_list_base[i].endswith(f"{i}.png") and img_list_trans[i].endswith(
            f"{i}.png"
        ):
            count += 1
        else:
            print(
                f"false: base({img_list_base[i][-8:]}) - trans({img_list_trans[i][-8:]})"
            )

    print(f"true: {count} total: {len(img_list_base)}")
    assert count == len(img_list_base)

    total_mse = 0.0
    for img_path_base, img_path_trans in tqdm(zip(img_list_base, img_list_trans)):
        img_base = Image.open(img_path_base)
        img_trans = Image.open(img_path_trans)

        mse = calculate_mse(img_base, img_trans)
        total_mse += mse

    average_mse = total_mse / N
    print(f"Average MSE: {average_mse}")
