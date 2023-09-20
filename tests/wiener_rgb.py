from skimage import io, color
from skimage.restoration import wiener
import numpy as np
import matplotlib.pyplot as plt

# image = io.imread(
#     # "tests/img/dog2.png",
#     "/home/tfukuda/project/guided-diffusion/results/classifier/classifier_sample_64_0/classifier_sample_64_1.png"
# )
# image = color.rgb2gray(image)
# print(image)
# psf = np.ones((5, 5)) / 25
# filtered_image = wiener(image, psf, 0.01)
# print(filtered_image)

# plt.imshow(filtered_image, cmap="gray")
# plt.show()


# 画像を読み込む
def save_lowfreq_image(input, output):
    image = io.imread(
        # "tests/img/dog2.png"
        # "/home/tfukuda/project/guided-diffusion/results/classifier/classifier_sample_64_0/classifier_sample_64_2.png",
        input, as_gray=False
    )

    image_np = image.copy().astype(np.float64)
    print(image_np.shape)
    if image_np.shape == (64, 64):
        # print('aaaaaaaaaaaaaaa')
        image_np = np.stack((image_np,) * 3, axis=-1)

    # 各チャンネルに分解する
    r, g, b = (image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2])

    r /= 255.0
    g /= 255.0
    b /= 255.0

    # 各チャンネルに対してウィナーフィルタを適用する
    psf = np.ones((5, 5)) / 25
    r_filtered = wiener(r, psf, 0.01)
    g_filtered = wiener(g, psf, 0.01)
    b_filtered = wiener(b, psf, 0.01)

    # 処理された画像を再構成する
    filtered_image = np.stack([r_filtered, g_filtered, b_filtered], axis=2)

    filtered_image = np.clip(filtered_image, 0.0, 1.0)
    # 処理された画像を表示する
    plt.imshow(filtered_image)
    # print(filtered_image)
    plt.imsave(output, filtered_image, format="jpeg")

    # print(r_filtered)
    # plt.imshow(g, cmap="gray")
    # plt.show()
