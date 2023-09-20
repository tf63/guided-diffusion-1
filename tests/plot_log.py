import matplotlib.pyplot as plt
import pandas as pd

# ログファイルのデータを読み込む
data = pd.read_csv("models/tiny_imagenet_cond/progress.csv")

# グラフのプロット
value = data["loss"]
# value = data["mse"]
# value = data["vb"]
plt.plot(data["step"][10:], value[10:], label="loss")
# plt.plot(data["step"], data["mse"], label="mse")
# plt.plot(data["step"], data["vb"], label="vb")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.show()
