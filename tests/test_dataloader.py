from guided_diffusion.image_datasets import load_data
from guided_diffusion.image_datasets import _list_image_files_recursively

data_dir = "/home/tfukuda/project/data/tiny-imagenet-200/train"
# data_dir = "/home/tfukuda/project/data/cifar10_64/cifar10-64/train"
print(_list_image_files_recursively(data_dir))

data = load_data(data_dir=data_dir, batch_size=8, image_size=64)
