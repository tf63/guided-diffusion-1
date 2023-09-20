import os
from gaussian import save_gaussian_image

data_dir = "/home/tfukuda/project/data/tiny-imagenet-200/train"
data_out_dir = "/home/tfukuda/project/data/tiny-imagenet-200_gaussian_k10_s10/train"
labels = [
    name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
]

for label in labels:
    label_dir = os.path.join(data_dir, label, "images")
    out_dir = os.path.join(data_out_dir, label, "images")
    os.makedirs(out_dir, exist_ok=True)
    image_files = [
        os.path.join(label_dir, f)
        for f in os.listdir(label_dir)
        if os.path.isfile(os.path.join(label_dir, f)) and f.endswith(".JPEG")
    ]

    for image_file in image_files:
        filename = os.path.basename(image_file)
        out_file = os.path.join(out_dir, filename)
        print(f"save image: {image_file}")
        save_gaussian_image(image_file, out_file)
