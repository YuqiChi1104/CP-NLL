import torch
import torchvision.transforms as transforms
from datasets import Dataset
import numpy as np
from PIL import Image
import os
import json
import model.model as module_arch
from parse_config import ConfigParser
import collections
import argparse
from sklearn.model_selection import train_test_split

# ==== 定义模型 ====
num_classes = 10
input_channel=3
gpuid = 1
checkpoint_path = "./ELR/model/cifar10lt200/lt200_0.9_150.pth"
data_dir = "./data/cifar-10-lt-200/cifar10-train.arrow"
save_dir = "./lt"
noise_file = "./ELR/noise/cifar10lt200/0.9_lt200_cifar10_sym_no.json"
device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")

args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
    CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')),
    CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')),
    CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
    CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
    CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
    CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
]
config = ConfigParser.get_instance(args, options)
model = config.initialize('arch', module_arch)  # 或者直接使用模型类初始化
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')  # 如果用GPU0
print(checkpoint.keys())
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()



# ==== 预处理 (和训练时一致) ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# ==== 保存路径 ====
os.makedirs(save_dir, exist_ok=True)

all_images, all_labels = [], []

# ==== 循环读取每个 batch ====
ds = Dataset.from_file(data_dir)
train_images = np.array(ds["img"])   # (N, 3072) or (N, 3, 32, 32)
train_images = train_images.transpose(0, 3, 1, 2)
train_labels = np.array(ds["label"]) # (N,)

def run_inference_and_save(model, images, labels, noise_labels, save_dir, batch_size=100):
    os.makedirs(save_dir, exist_ok=True)
    model_results = []
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
        imgs_tensor = torch.cat(imgs_tensor, dim=0).to(device) 

        # 推理
        with torch.no_grad():
            outputs = model(imgs_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # ==== 保存为 [noise_label, scores] ====
        model_results.extend([[tl, nl, s.tolist()] for tl, nl, s in zip(batch_labels, batch_noise_labels, probs)])


        processed_samples += imgs_tensor.size(0)
        percent = (processed_samples / total_samples) * 100
        print(f"\r已完成: {processed_samples}/{total_samples} ({percent:.2f}%)", end='')

    # ==== 保存为 npy ====
    np.save(os.path.join(save_dir, "model_scores.npy"), np.array(model_results, dtype=object))

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
    run_inference_and_save(model, val_images, val_labels, val_noise, os.path.join(save_dir, save_root, "val"))
    run_inference_and_save(model, test_images, test_labels, test_noise, os.path.join(save_dir, save_root, "test"))
