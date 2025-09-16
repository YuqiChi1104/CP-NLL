import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
import os
import json
from model.SevenCNN import CNN
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import json
import torch
import torch.nn as nn
from utils import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from model.SevenCNN import CNN
from data.imbalance_cifar import *
from data.Clothing1M import *
import argparse
from datasets import Dataset
# ==== 定义模型 ====
n_classes=10
checkpoint_path = ".l/SED/model/cifar10lt200/cifar10lt200_0.9_30.pth"
noise_file = "./SED/noise/cifar10lt200/cifar10lt200_symmetric_0.9.json"
checkpoint = torch.load(checkpoint_path)

# ==== CIFAR-10 数据路径 ====
data_dir = "./data/cifar-10-lt-200/cifar10-train.arrow"
# ==== 定义模型 ====
class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.proba_head = torch.nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}
def build_model(num_classes, params_init, dev, config):
    if config.dataset.startswith('web-'):
        net = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
    else:
        net = CNN(input_channel=3, n_outputs=n_classes)

    return net.to(dev)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr-decay', type=str, default='cosine:20,5e-4,100')
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--opt', type=str, default='sgd')
parser.add_argument('--warmup-epochs', type=int, default=20)
parser.add_argument('--warmup-lr', type=float, default=0.001)
parser.add_argument('--warmup-gradual', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--params-init', type=str, default='none')

parser.add_argument('--aph', type=float, default=0.95)

parser.add_argument('--dataset', type=str, default='cifar100nc')
parser.add_argument('--noise-type', type=str, default='symmetric')
parser.add_argument('--closeset-ratio', type=float, default=0.2)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--save-weights', type=bool, default=False)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--restart-epoch', type=int, default=0)

parser.add_argument('--use-quantile', type=bool, default=True)
parser.add_argument('--clip-thresh', type=bool, default=True)
parser.add_argument('--use-mixup', type=bool, default=False)
parser.add_argument('--momentum_scs', type=float, default=0.999)
parser.add_argument('--momentum_scr', type=float, default=0.999)
parser.add_argument('--imb_factor', type=int, default=50)


args = parser.parse_args()

config = args
device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
model = build_model(n_classes, config.params_init, device, config)

print(type(checkpoint))   # <class 'dict'>
print(checkpoint.keys())  # dict_keys(['net_state_dict', 'optimizer_state_dict', 'epoch', ...])

model.load_state_dict(checkpoint)



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
            logits = outputs['logits']    
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # ==== 保存为 [noise_label, scores] ====
        model_results.extend([[tl, nl, s.tolist()] for tl, nl, s in zip(batch_labels, batch_noise_labels, probs)])


        processed_samples += imgs_tensor.size(0)
        percent = (processed_samples / total_samples) * 100
        print(f"\r已完成: {processed_samples}/{total_samples} ({percent:.2f}%)", end='')

    # ==== 保存为 npy ====
    np.save(os.path.join(save_dir, "net_scores.npy"), np.array(model_results, dtype=object))

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
