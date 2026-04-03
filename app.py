# -*- coding: utf-8 -*-
"""
Flask 后端：加载三种模型（TwoLayerNet / SimpleCNN / LeNet5），统一推理接口。

画布图像经 preprocess_canvas 做「反色 + 居中方形裁剪 + 20→28 嵌入」后再推理，
与 MNIST 训练分布对齐。
"""

from __future__ import annotations

import base64
import io
import os

# Windows 下 NumPy/MKL 与 PyTorch 重复加载 OpenMP 时的规避（须先于 numpy/torch）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image

from preprocess_canvas import (
    gray01_to_torch_nchw,
    gray01_to_two_layer_flat,
    is_mostly_blank,
    preprocess_pil_to_mnist_gray01,
)
from train import LeNet5, SimpleCNN
from two_layer_net import TwoLayerNet

# -----------------------------------------------------------------------------
# 路径与设备
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

_two_layer: Optional[TwoLayerNet] = None
_cnn: Optional[SimpleCNN] = None
_lenet: Optional[LeNet5] = None


def _load_state_dict_flexible(model: torch.nn.Module, path: str) -> None:
    """兼容不同 PyTorch 版本的 torch.load 参数。"""
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)


def load_all_models() -> None:
    global _two_layer, _cnn, _lenet

    p_npz = os.path.join(WEIGHTS_DIR, "two_layer.npz")
    if os.path.isfile(p_npz):
        data = np.load(p_npz, allow_pickle=False)
        in_dim, hidden = int(data["W1"].shape[0]), int(data["W1"].shape[1])
        _two_layer = TwoLayerNet(
            input_size=in_dim, hidden_size=hidden, output_size=10, weight_init_std=0.01
        )
        _two_layer.load_params(p_npz)
        print(f"[OK] TwoLayerNet 已加载: {p_npz} (hidden={hidden})")
    else:
        print(f"[警告] 未找到 {p_npz}，请先运行 python train.py 训练 NumPy 模型")

    p_cnn = os.path.join(WEIGHTS_DIR, "cnn.pth")
    if os.path.isfile(p_cnn):
        _cnn = SimpleCNN()
        _load_state_dict_flexible(_cnn, p_cnn)
        _cnn.to(DEVICE)
        _cnn.eval()
        print(f"[OK] SimpleCNN 已加载: {p_cnn}")
    else:
        print(f"[警告] 未找到 {p_cnn}，请先训练 CNN")

    p_lenet = os.path.join(WEIGHTS_DIR, "lenet.pth")
    if os.path.isfile(p_lenet):
        _lenet = LeNet5()
        _load_state_dict_flexible(_lenet, p_lenet)
        _lenet.to(DEVICE)
        _lenet.eval()
        print(f"[OK] LeNet5 已加载: {p_lenet}")
    else:
        print(f"[警告] 未找到 {p_lenet}，请先训练 LeNet")


def decode_base64_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


def predict_two_layer(x_784: np.ndarray) -> Tuple[int, float, np.ndarray]:
    assert _two_layer is not None
    probs = _two_layer.predict(x_784)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs


def predict_torch(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    x = x.to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs_t = F.softmax(logits, dim=1)[0]
    probs = probs_t.cpu().numpy()
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs


@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "无效的 JSON"}), 400

    model_key = payload.get("model", "two_layer")
    image_data = payload.get("image", "")
    if not image_data:
        return jsonify({"error": "缺少 image 字段"}), 400

    try:
        pil_img = decode_base64_image(image_data)
    except Exception as e:
        return jsonify({"error": f"图片解析失败: {e}"}), 400

    arr28 = preprocess_pil_to_mnist_gray01(pil_img, assume_white_background=True)
    if is_mostly_blank(arr28):
        return jsonify({"error": "画布几乎为空，请先书写数字"}), 400

    tensor = gray01_to_torch_nchw(arr28)

    if model_key == "two_layer":
        if _two_layer is None:
            return jsonify({"error": "TwoLayerNet 未加载，请先运行 train.py"}), 503
        x_np = gray01_to_two_layer_flat(arr28)
        digit, confidence, probs = predict_two_layer(x_np)
    elif model_key == "cnn":
        if _cnn is None:
            return jsonify({"error": "CNN 未加载，请先运行 train.py"}), 503
        digit, confidence, probs = predict_torch(_cnn, tensor)
    elif model_key == "lenet":
        if _lenet is None:
            return jsonify({"error": "LeNet 未加载，请先运行 train.py"}), 503
        digit, confidence, probs = predict_torch(_lenet, tensor)
    else:
        return jsonify({"error": f"未知模型: {model_key}"}), 400

    return jsonify(
        {
            "digit": digit,
            "confidence": round(confidence, 6),
            "probabilities": [round(float(p), 6) for p in probs.tolist()],
        }
    )


load_all_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
