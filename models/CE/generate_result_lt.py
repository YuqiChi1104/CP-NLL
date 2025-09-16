import torch
import torchvision.transforms as transforms
import numpy as np
from PreResNet import ResNet18
from PIL import Image
from datasets import Dataset
import os
import json
from sklearn.model_selection import train_test_split

# ==== 定义模型 ====
num_classes = 10
net = ResNet18(num_classes=num_classes).cuda()
net.eval()

# ==== 加载权重 ====
checkpoint_path = "./CE/model/cifar10lt200/CE_cifar10lt200_r0.9_epoch300.pth"
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint["net_state_dict"])

# ==== CIFAR-10 数据路径 ====
data_dir = "./data/cifar-10-lt-200/cifar10-train.arrow"
noise_file = "./CE/noise/cifar10lt200/CE_cifar10lt200_0.90.json"


# ==== 预处理 (和训练时一致) ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# ==== 保存路径 ====
save_dir = "./lt"
os.makedirs(save_dir, exist_ok=True)

all_images, all_labels = [], []


# ==== 循环读取每个 batch ====
ds = Dataset.from_file(data_dir)
train_images = np.array(ds["img"])   # (N, 3072) or (N, 3, 32, 32)
train_images = train_images.transpose(0, 3, 1, 2)
train_labels = np.array(ds["label"]) # (N,)

def run_inference_and_save(net, images, labels, noise_labels, save_dir, batch_size=100):
    os.makedirs(save_dir, exist_ok=True)
    net_results = []
    total_samples, processed_samples = images.shape[0], 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        imgs = images[start_idx:end_idx]
        batch_noise_labels = noise_labels[start_idx:end_idx]  # 对应噪音标签
        batch_labels = labels[start_idx:end_idx]

        # 转 tensor
        imgs_tensor = []
        for img in imgs:
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0)  # [1,3,32,32]
            imgs_tensor.append(img)
        imgs_tensor = torch.cat(imgs_tensor, dim=0).cuda()

        # 推理
        with torch.no_grad():
            outputs = net(imgs_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # ==== 保存为 [noise_label, scores] ====
        net_results.extend([[tl, nl, s.tolist()] for tl, nl, s in zip(batch_labels, batch_noise_labels, probs)])

        processed_samples += imgs_tensor.size(0)
        percent = (processed_samples / total_samples) * 100
        print(f"\r已完成: {processed_samples}/{total_samples} ({percent:.2f}%)", end='')

    # ==== 保存为 npy ====
    np.save(os.path.join(save_dir, "net_scores.npy"), np.array(net_results, dtype=object))

    print(f"\n保存完成！文件路径: {save_dir}")

# 载入 noise_labels

with open(noise_file, "r") as f:
    noise_labels = json.load(f)

print("images:", train_images.shape)        # (50000, 3, 32, 32)
print("labels:", train_labels.shape)        # (50000,)
print("noise_labels:", len(noise_labels))  # (50000,)

val_images, test_images, val_labels, test_labels, val_noise, test_noise = train_test_split(
    train_images, train_labels, noise_labels, test_size=0.4, random_state=42, stratify=train_labels
)

# 保存到 scores1 / scores2
for save_root in ["scores1", "scores2"]:
    run_inference_and_save(net, val_images, val_labels, val_noise, os.path.join(save_dir, save_root, "val"))
    run_inference_and_save(net, test_images, test_labels, test_noise, os.path.join(save_dir, save_root, "test"))
