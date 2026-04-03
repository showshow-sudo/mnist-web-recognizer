# -*- coding: utf-8 -*-
"""
训练脚本：
1. 鱼书 TwoLayerNet（NumPy）在 MNIST 上训练并保存 weights/two_layer.npz
2. 简单 CNN（PyTorch）保存 weights/cnn.pth
3. 经典 LeNet-5 风格网络（PyTorch）保存 weights/lenet.pth

在项目根目录执行: python train.py

训练目标（MNIST 官方测试集，与 transforms.ToTensor 一致）：
- TwoLayerNet（隐藏层加大 + 更长训练 + 学习率衰减）：约 96%+
- CNN / LeNet：约 99%+
"""

from __future__ import annotations

import os

# Windows 常见：PyTorch 与 NumPy/Intel-MKL 各自链接 OpenMP，会触发
# "libiomp5md.dll already initialized"。须在导入 numpy/torch 之前设置。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from two_layer_net import TwoLayerNet

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """MNIST：ToTensor → [0,1]，单通道，**与 preprocess_canvas 推理预处理一致**。"""
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=os.path.join(ROOT, "data"),
        train=True,
        download=True,
        transform=tfm,
    )
    test_ds = datasets.MNIST(
        root=os.path.join(ROOT, "data"),
        train=False,
        download=True,
        transform=tfm,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_two_layer_net(
    epochs: int = 35,
    batch_size: int = 128,
    lr_init: float = 0.1,
    lr_gamma: float = 0.97,
    hidden_size: int = 256,
) -> None:
    """NumPy TwoLayerNet + mini-batch SGD；学习率按 epoch 指数衰减以稳住后期收敛。"""
    print("=" * 60)
    print("[1/3] 训练 TwoLayerNet (NumPy / 鱼书)")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(batch_size)
    net = TwoLayerNet(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        weight_init_std=0.01,
    )

    def accuracy(x_np: np.ndarray, y_np: np.ndarray) -> float:
        y_pred = np.argmax(net.predict(x_np), axis=1)
        return float(np.mean(y_pred == y_np))

    train_x_list, train_y_list = [], []
    for imgs, labels in train_loader:
        train_x_list.append(imgs.numpy().reshape(-1, 784))
        train_y_list.append(labels.numpy())
    x_all = np.concatenate(train_x_list, axis=0)
    y_all = np.concatenate(train_y_list, axis=0)
    train_size = x_all.shape[0]

    test_x_list, test_y_list = [], []
    for imgs, labels in test_loader:
        test_x_list.append(imgs.numpy().reshape(-1, 784))
        test_y_list.append(labels.numpy())
    x_test = np.concatenate(test_x_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)

    iters_per_epoch = max(train_size // batch_size, 1)

    for epoch in range(epochs):
        lr = lr_init * (lr_gamma**epoch)
        perm = np.random.permutation(train_size)
        sum_loss = 0.0
        for i in range(iters_per_epoch):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x_batch = x_all[idx]
            y_batch = y_all[idx]

            loss, grads = net.loss_and_grad(x_batch, y_batch)
            sum_loss += loss
            for key in ("W1", "b1", "W2", "b2"):
                net.params[key] -= lr * grads[key]

        avg_loss = sum_loss / iters_per_epoch
        acc_tr = accuracy(x_all, y_all)
        acc_te = accuracy(x_test, y_test)
        print(
            f"  Epoch {epoch + 1}/{epochs}  lr={lr:.5f}  loss={avg_loss:.4f}  "
            f"train_acc={acc_tr:.4f}  test_acc={acc_te:.4f}"
        )

    print(f"  [最终] TwoLayerNet 测试集准确率 ≈ {accuracy(x_test, y_test) * 100:.2f}%")

    out_path = os.path.join(WEIGHTS_DIR, "two_layer.npz")
    net.save_params(out_path)
    print(f"  已保存: {out_path}\n")


class SimpleCNN(nn.Module):
    """简单 CNN：Conv-BN-ReLU-Pool ×2 + 全连接。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    """
    LeNet-5 结构适配 28×28 MNIST：
    C1(5,pad=2)→S2→C3(5)→S4→FC 120→84→10，Tanh + AvgPool。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_pytorch_model(
    model: nn.Module,
    name: str,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    print("=" * 60)
    print(f"训练 PyTorch 模型: {name}")
    print("=" * 60)

    model = model.to(DEVICE)
    train_loader, test_loader = get_mnist_loaders(batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(
            f"  Epoch {epoch + 1}/{epochs}  "
            f"train_loss={total_loss / max(n_batches, 1):.4f}  test_acc={acc:.4f}"
        )

    print(f"  [本模型最佳] 测试集准确率 ≈ {best_acc * 100:.2f}%")

    save_path = os.path.join(WEIGHTS_DIR, f"{name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  已保存: {save_path}\n")


def main() -> None:
    print(f"使用设备: {DEVICE}\n")
    train_two_layer_net(
        epochs=35,
        batch_size=128,
        lr_init=0.1,
        lr_gamma=0.97,
        hidden_size=256,
    )
    train_pytorch_model(SimpleCNN(), "cnn", epochs=15, batch_size=128, lr=1e-3)
    train_pytorch_model(LeNet5(), "lenet", epochs=22, batch_size=128, lr=1e-3)
    print("全部训练完成。权重位于目录:", WEIGHTS_DIR)


if __name__ == "__main__":
    main()
