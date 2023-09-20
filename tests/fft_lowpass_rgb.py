import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
img = cv2.imread("tests/img/dog2.png")

# RGB成分に分解する
b, g, r = cv2.split(img)

# 各チャンネルごとにフーリエ変換する
b_f = np.fft.fft2(b)
b_fshift = np.fft.fftshift(b_f)

g_f = np.fft.fft2(g)
g_fshift = np.fft.fftshift(g_f)

r_f = np.fft.fft2(r)
r_fshift = np.fft.fftshift(r_f)

# 振幅スペクトルを取得する
magnitude_spectrum_b = 20 * np.log(np.abs(b_fshift))
magnitude_spectrum_g = 20 * np.log(np.abs(g_fshift))
magnitude_spectrum_r = 20 * np.log(np.abs(r_fshift))

# 位相スペクトルを取得する
phase_spectrum_b = np.angle(b_fshift)
phase_spectrum_g = np.angle(g_fshift)
phase_spectrum_r = np.angle(r_fshift)

print(b_fshift.shape)
# 各チャンネルに対してローパスフィルタをかける
rows, cols = img.shape[:2]
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 60  # ローパスフィルタの半径
center = [crow, ccol]  # 中心座標
x, y = np.ogrid[:rows, :cols]
dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)  # 中心からの距離
mask = (dist_from_center < r).astype(np.uint8)  # ローパスフィルタの作成
# mask[..., 1] = mask[..., 0]
# print(mask.shape)
# print(mask)

b_fshift *= mask
g_fshift *= mask
r_fshift *= mask

# 逆フーリエ変換して画像を復元する
b_ishift = np.fft.ifftshift(b_fshift)
b_back = np.fft.ifft2(b_ishift)
b_back = np.abs(b_back)

g_ishift = np.fft.ifftshift(g_fshift)
g_back = np.fft.ifft2(g_ishift)
g_back = np.abs(g_back)

r_ishift = np.fft.ifftshift(r_fshift)
r_back = np.fft.ifft2(r_ishift)
r_back = np.abs(r_back)

# ローパスフィルタを表示する
mask *= 255
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
# mask_rgb[..., 0] = 0
# mask_rgb[..., 2] = 0

# 各チャンネルを結合して、フィルタ処理後の画像を作成する
img_back = cv2.merge((b_back, g_back, r_back))

# 元画像とスペクトル、フィルタ処理後の画像を表示する
plt.subplot(3, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Input Image"), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 2), plt.imshow(mask_rgb)
plt.title("Low Pass Filter"), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 3), plt.imshow(
    cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_BGR2RGB)
)
plt.title("Output Image"), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 4), plt.imshow(magnitude_spectrum_r, cmap="gray")
plt.title("Magnitude (R)"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(magnitude_spectrum_g, cmap="gray")
plt.title("Magnitude (G)"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(magnitude_spectrum_b, cmap="gray")
plt.title("Magnitude (B)"), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 7), plt.imshow(phase_spectrum_r, cmap="gray")
plt.title("Phase (R)"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(phase_spectrum_g, cmap="gray")
plt.title("Phase (G)"), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(phase_spectrum_b, cmap="gray")
plt.title("Phase (B)"), plt.xticks([]), plt.yticks([])

plt.tight_layout()

plt.savefig("tests/out/rgb.png")
# plt.show()
