# -*- coding: utf-8 -*-
"""
两层神经网络 TwoLayerNet（《深度学习入门：基于 Python 的理论与实现》/ 鱼书风格）

结构：Affine -> ReLU -> Affine -> Softmax（分类）
仅依赖 NumPy，可用于训练与推理。
"""

from typing import Dict, Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Softmax（按最后一维）。"""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    多分类交叉熵。y_true 为 one-hot 或整型标签 (batch,)。
    y_pred: (batch, C) 概率或 logits（此处为 softmax 后概率）。
    """
    if y_true.ndim == 1:
        batch_size = y_pred.shape[0]
        return -np.mean(np.log(y_pred[np.arange(batch_size), y_true] + 1e-8))
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def relu_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = x > 0
    out = x.copy()
    out[~mask] = 0
    return out, mask


def relu_backward(dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dx = dout.copy()
    dx[~mask] = 0
    return dx


def affine_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, Tuple]:
    """全连接前向。x: (N, D)"""
    out = x @ w + b
    cache = (x, w, b)
    return out, cache


def affine_backward(
    dout: np.ndarray, cache: Tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w, _ = cache
    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db


def softmax_with_loss_forward(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    最后一层：Softmax + 交叉熵，合并计算反向传播时梯度更简单。
    y: one-hot 或整型类别索引 (batch,)。
    返回 loss 与 对 x 的梯度（在 backward 里用）。
    """
    prob = softmax(x)
    loss = cross_entropy_error(prob, y)

    batch_size = x.shape[0]
    if y.ndim == 1:
        dx = prob.copy()
        dx[np.arange(batch_size), y] -= 1
        dx /= batch_size
    else:
        dx = (prob - y) / batch_size
    return loss, dx


class TwoLayerNet:
    """
    鱼书标准两层网络：input_size -> hidden_size -> output_size
    参数放在 self.params 字典中：W1, b1, W2, b2
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 100,
        output_size: int = 10,
        weight_init_std: float = 0.01,
    ):
        self.params: Dict[str, np.ndarray] = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """前向推理，返回各类别概率 (N, C)。x 可为 (N, 784) 或展平前形状在外部处理。"""
        w1, b1, w2, b2 = (
            self.params["W1"],
            self.params["b1"],
            self.params["W2"],
            self.params["b2"],
        )
        a1 = x @ w1 + b1
        z1 = np.maximum(0, a1)  # ReLU
        a2 = z1 @ w2 + b2
        return softmax(a2)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """训练用：前向 + Softmax+交叉熵损失。"""
        w1, b1, w2, b2 = (
            self.params["W1"],
            self.params["b1"],
            self.params["W2"],
            self.params["b2"],
        )
        # Affine1 -> ReLU -> Affine2
        a1, cache1 = affine_forward(x, w1, b1)
        z1, relu_mask = relu_forward(a1)
        a2, cache2 = affine_forward(z1, w2, b2)
        loss, dout = softmax_with_loss_forward(a2, y)

        # 反向
        dz1, dw2, db2 = affine_backward(dout, cache2)
        da1 = relu_backward(dz1, relu_mask)
        _, dw1, db1 = affine_backward(da1, cache1)
        self._last_grads = {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}
        return loss

    def gradient(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """计算当前 batch 的梯度（内部调用一次 loss）。"""
        self.loss(x, y)
        return self._last_grads

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """一次前向+反向，返回 (loss, grads)，避免训练循环重复计算。"""
        loss = self.loss(x, y)
        return float(loss), self._last_grads

    def load_params(self, path: str) -> None:
        """从 .npz 加载参数。"""
        data = np.load(path, allow_pickle=False)
        for k in ("W1", "b1", "W2", "b2"):
            self.params[k] = data[k]

    def save_params(self, path: str) -> None:
        """保存为 .npz。"""
        np.savez(
            path,
            W1=self.params["W1"],
            b1=self.params["b1"],
            W2=self.params["W2"],
            b2=self.params["b2"],
        )
