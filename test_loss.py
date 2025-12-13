import matplotlib.pyplot as plt

# # 模型及对应最终Loss
# models = ["STGCN", "MTGNN", "Graph WaveNet"]
# losses = [0.174320, 0.159851, 0.129249]
#
# # 绘制柱状图
# plt.figure(figsize=(6,5))
# bars = plt.bar(models, losses, color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
#
# # 在柱顶显示数值
# for bar, loss in zip(bars, losses):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
#              f"{loss:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
#
# # 美化图像
# plt.ylabel("Test Loss (MSE)", fontsize=12)
# plt.title("Loss", fontsize=14, fontweight='bold')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
#
# # 保存图片
# plt.savefig("/data/zsm/case01/figures/final_loss_comparison.png", dpi=300)
# plt.show()

import numpy as np


def plot_loss_curves(loss_dict, save_path="loss_comparison.png"):
    """
    绘制多个模型的验证损失下降曲线

    参数：
        loss_dict (dict): { "ModelName": [loss1, loss2, ...], ... }
        save_path (str): 保存路径
    """
    plt.figure(figsize=(8, 5))
    for model_name, losses in loss_dict.items():
        plt.plot(losses, label=model_name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

loss_dict = {
    "STGCN": np.load("/data/zsm/case01/models/stgcn_best_val_loss.npy").tolist(),
    "MTGNN": np.load("/data/zsm/case01/model_saved/MTGNN/best_model_val_loss.npy").tolist(),
    "GWN": np.load("/data/zsm/case01/models/gwn_best_val_loss.npy").tolist(),
}

plot_loss_curves(loss_dict, save_path="loss_comparison.png")