from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# モデルをダウンロードします。
model = models.inception_v3(pretrained=True)

# モデルをロードし、GPU に転送します。
# model.load_state_dict(torch.load("inception_v3.pth"))
model.cuda()

# モデルに飛行機の画像を入力します。
image = Image.open(
    "results/trans/cg64_dir_404to817_t500_run0/cg64_dir_404to817_t500_0.png"
).convert("RGB")
# image = torch.from_numpy(np.array(image)).float()
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(299),
    ]
)

image = preprocess(image)
image = image.unsqueeze(0)
image = image.cuda()

# モデルの出力を取得します。
output = model(image)

# 出力をソフトマックス関数に通します。
softmax = nn.Softmax(dim=1)
output = softmax(output)

# ソフトマックス関数の出力を取得します。
probabilities = output.cpu().data.numpy()[0]

# ソフトマックス関数の出力を最大化するクラスのインデックスを見つけます。
max_index = np.argmax(probabilities)

# 最大化するクラスのインデックスが飛行機のクラスのインデックスと等しいかどうかを確認します。
if max_index == 283:
    print("画像は飛行機である可能性が高いです。")
else:
    print("画像は飛行機である可能性は低いです。")
