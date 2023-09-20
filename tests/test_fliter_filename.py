# import os

# img_dir = "results/image/tiny_sample_cond500000_gaussian_label0_run_0"
# img_list = []

# for root, dirs, files in os.walk(img_dir):
#     for filename in files:
#         if filename.endswith(".png") and not filename.endswith("_list.png"):
#             file_path = os.path.join(root, filename)
#             img_list.append(file_path)

# img_list = sorted(img_list)
# print(img_list)

import os
from natsort import natsorted

img_dir = "results/image/tiny_sample_cond500000_gaussian_label0_run_0"
img_list = []

for filename in natsorted(os.listdir(img_dir)):
    if filename.endswith(".png") and not filename.endswith("_list.png"):
        file_path = os.path.join(img_dir, filename)
        img_list.append(file_path)

print(img_list)
