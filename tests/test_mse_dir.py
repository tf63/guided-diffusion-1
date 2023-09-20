import os
from natsort import natsorted


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


img_dir_base = "results/image/tcfg64_label404_run0"
img_list_base = get_natsorted_img_list(img_dir_base)
img_dir_trans = "results/trans/trans64_dir_404to817_t500_run0"
img_list_trans = get_natsorted_img_list(img_dir_trans)

count = 0
for i in range(9600):
    if img_list_base[i].endswith(f"{i}.png") and img_list_trans[i].endswith(f"{i}.png"):
        # if img_list_base[i][-8:] == img_list_trans[i][-8:]:
        count += 1
    else:
        print(f"false: {img_list_base[i][-8:]} - {img_list_trans[i][-8:]}")
print(f"count: {count} length: {len(img_list_base)}")
