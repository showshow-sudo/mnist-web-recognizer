# MNIST 手写数字识别 Web

基于浏览器画布的 **MNIST 手写数字识别** 演示项目：Flask 后端 + HTML/CSS/JavaScript 前端，集成 **三种模型**（鱼书 TwoLayerNet / 简单 CNN / LeNet-5 风格网络），统一推理接口与画布图像预处理。

## 功能概览

- 鼠标或触控在画布上手写 0–9，选择模型后点击**识别**，返回预测数字、置信度与 10 类概率。
- **三种模型可切换**：NumPy 两层全连接（鱼书风格）、PyTorch CNN、PyTorch LeNet-5。
- **预处理**：将白底黑字的画布图对齐为 MNIST 风格（灰度、反色、包围盒裁切、方形扩边、缩放至 20×20 后嵌入 28×28、像素归一化到 [0,1]），与 `torchvision` 的 `ToTensor()` 训练管道一致。

## 技术栈

| 类别 | 说明 |
|------|------|
| 后端 | Python 3.9+、Flask |
| 数值 / 深度学习 | NumPy、PyTorch、torchvision |
| 图像 | Pillow |
| 前端 | HTML、CSS、JavaScript（Canvas） |

## 目录结构

```
├── app.py                 # Flask 应用：加载模型、/predict 接口
├── train.py               # 训练三种模型并保存权重
├── two_layer_net.py       # 鱼书风格 TwoLayerNet（NumPy）
├── preprocess_canvas.py   # 画布图 → MNIST 风格张量
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/canvas.js
```

训练/下载生成的目录（默认不纳入 Git）：

- `data/`：MNIST 数据集缓存  
- `weights/`：`two_layer.npz`、`cnn.pth`、`lenet.pth`

## 环境安装

```bash
# 建议使用虚拟环境
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **Windows 提示**：若出现 OpenMP / `libiomp5md.dll` 相关警告，仓库内已在 `train.py` / `app.py` 中设置 `KMP_DUPLICATE_LIB_OK`（见源码）。若仍异常，可检查本机 PyTorch 与 NumPy 安装来源是否混用。

## 训练模型

在项目根目录执行（将下载 MNIST 至 `data/`，并将权重写入 `weights/`）：

```bash
python train.py
```

完成后应生成：

- `weights/two_layer.npz`
- `weights/cnn.pth`
- `weights/lenet.pth`

未训练直接启动网站时，对应模型将无法加载，接口会返回提示信息。

## 运行网站

```bash
python app.py
```

浏览器访问：**http://127.0.0.1:5000**

## API 说明（简要）

- **POST** `/predict`  
  - Content-Type: `application/json`  
  - 字段：`model` 取 `two_layer` | `cnn` | `lenet`；`image` 为 PNG 的 Base64 Data URL（与前端 `canvas.toDataURL('image/png')` 一致）。  
  - 成功时返回：`digit`、`confidence`、`probabilities`（长度 10）。

## 许可证

本项目仅供学习与交流使用。MNIST 数据集版权归原始发布方所有。
