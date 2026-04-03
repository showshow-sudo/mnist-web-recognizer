# -*- coding: utf-8 -*-
"""
浏览器画布图像 → MNIST 风格张量/矩阵的预处理。

MNIST 官方字符约占约 20×20 像素并被置于 28×28 黑底画布中央。
若直接把整张大画布线性缩放成 28×28，笔画相对整张图过小，模型看到的主要
是「大片空白」，与训练集分布严重不符，识别会极差。

本模块顺序：
1. 灰度 + 反色：白底黑字 → 黑底白字（与 torchvision MNIST + ToTensor 一致）
2. 按笔画找包围盒，扩边后裁成**居中方形**（等比例保留字形）
3. 将方形区域缩放到 20×20，再**嵌入** 28×28 黑底中心（与 MNIST 留白一致）
4. 数值约束在 [0, 1]，与训练时 transforms.ToTensor() 一致（未做 dataset 级 Normalize）
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

# MNIST 中数字典型尺寸约 20 像素，四边各留约 4 像素边距
MNIST_CONTENT_SIZE = 20
OUTPUT_SIZE = 28


def preprocess_pil_to_mnist_gray01(
    pil_img: Image.Image,
    *,
    assume_white_background: bool = True,
) -> np.ndarray:
    """
    将 PIL 图像转为 float32、形状 (28,28)、取值 [0,1]、**白字黑底**（与 MNIST+ToTensor 一致）。

    :param pil_img: RGBA/RGB 等任意模式
    :param assume_white_background: True 表示白底黑字画布（本站 canvas 固定如此）
    """
    gray_uint8 = np.asarray(pil_img.convert("L"), dtype=np.float32)
    gray01 = gray_uint8 / 255.0

    if assume_white_background:
        # 黑字(低)白底(高) → MNIST 白字(高)黑底(低)
        digit = 1.0 - gray01
    else:
        digit = gray01

    h, w = digit.shape
    if h < 2 or w < 2:
        return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

    # 自适应阈值：避免抗锯齿弱笔画全部被截断
    fg = float(digit.max())
    if fg < 0.02:
        return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    thresh = max(0.05, fg * 0.12)

    ys, xs = np.where(digit > thresh)
    if ys.size == 0:
        return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    pad = int(round(max(box_h, box_w) * 0.22)) + 6

    y0p = max(0, y0 - pad)
    y1p = min(h - 1, y1 + pad)
    x0p = max(0, x0 - pad)
    x1p = min(w - 1, x1 + pad)

    crop = digit[y0p : y1p + 1, x0p : x1p + 1].copy()
    ch, cw = crop.shape
    side = max(ch, cw)

    # 居中放入方形画布，再整体等比例缩放到 20×20，嵌入 28×28
    square = np.zeros((side, side), dtype=np.float32)
    yo = (side - ch) // 2
    xo = (side - cw) // 2
    square[yo : yo + ch, xo : xo + cw] = crop

    pil_sq = Image.fromarray(np.clip(square * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    pil_20 = pil_sq.resize(
        (MNIST_CONTENT_SIZE, MNIST_CONTENT_SIZE),
        Image.Resampling.LANCZOS,
    )
    small = np.asarray(pil_20, dtype=np.float32) / 255.0

    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    off = (OUTPUT_SIZE - MNIST_CONTENT_SIZE) // 2
    out[off : off + MNIST_CONTENT_SIZE, off : off + MNIST_CONTENT_SIZE] = small
    return np.clip(out, 0.0, 1.0)


def gray01_to_torch_nchw(arr28: np.ndarray) -> torch.Tensor:
    """(28,28) numpy → (1,1,28,28) float32 CPU tensor"""
    t = torch.from_numpy(arr28).unsqueeze(0).unsqueeze(0)
    return t.contiguous()


def gray01_to_two_layer_flat(arr28: np.ndarray) -> np.ndarray:
    """(28,28) → (1,784) C 连续、行优先，与 MNIST 展平顺序一致"""
    flat = np.ascontiguousarray(arr28.reshape(1, -1), dtype=np.float32)
    return flat


def is_mostly_blank(arr28: np.ndarray, *, max_fg: float = 0.03) -> bool:
    """判断是否接近空白画布（避免误识别空图）。"""
    return float(arr28.max()) < max_fg
