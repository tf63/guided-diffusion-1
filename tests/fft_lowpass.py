import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
img = cv2.imread("tests/img/dog2.png", 0)

# 画像をフーリエ変換する
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 振幅スペクトルを取得する
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 位相スペクトルを取得する
phase_spectrum = np.angle(fshift)
print(img.shape)
# ローパスフィルタを作成する
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
c = 32
mask = np.zeros((rows, cols), np.uint8)
mask[crow - c : crow + c, ccol - c : ccol + c] = 1

# フーリエ変換した画像にローパスフィルタをかける
fshift = fshift * mask

# 逆フーリエ変換して画像を復元する
ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(ishift)
img_back = np.abs(img_back)

# 画像とスペクトルを表示する
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Input Image")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(phase_spectrum, cmap="gray")
plt.title("Phase Spectrum")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap="gray")
plt.title("Output Image")
plt.xticks([])
plt.yticks([])

plt.savefig("tests/out/out.png")
