import cv2
import numpy as np


def save_gaussian_image(input, output):
    # 画像を読み込む
    img = cv2.imread(input)

    # ガウシアンカーネルを作成する
    kernel_size = 10
    sigma = 10
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())

    # ガウシアンフィルタを適用する
    filtered_img = cv2.filter2D(img, -1, kernel)

    # フィルター処理された画像を保存する
    cv2.imwrite(output, filtered_img)
