## 目的：
从立体相机（左右两路图像）中，自动定位手术缝合针的三维抓取点（midpoint），用于机器人辅助手术中的自动抓针。
流程（7步）：
1. YOLO + SAM3：分别对左右图像做目标检测 + 实例分割，得到针的像素mask 
2. 骨架提取：对mask做细化（skeletonize），得到有序的1D曲线点列 
3. 去畸变：用相机标定参数消除镜头畸变 
4. 对极匹配：基于极线约束，将左右骨架点配对 
5. 三角化：将匹配点对三角化，重建3D点云 
6. 3D圆弧拟合：用RANSAC拟合3D圆（针是圆弧形），得到圆心、半径、法向量 
7. 圆弧中点：找圆弧的角度中点，作为最终的3D抓取目标点（输出坐标单位：mm） 

## 输出目录：outputs_3d

运行 `run_needle_3d.sh` 后，所有结果保存在 `outputs_3d/`（可通过 `--output-dir` 修改）：

| 文件 | 说明 |
|---|---|
| `needle_3d.npz` | 主结果文件，包含 3D 抓取点坐标及圆弧拟合参数（见下表） |
| `img_XXXX_result.jpg` | 左右图像上的 YOLO 检测框 + SAM3 分割 mask 叠加结果 |
| `skeleton.png` | 左右图像骨架提取结果可视化 |
| `epipolar.png` | 对极匹配结果可视化（左右骨架匹配点对及极线） |
| `reprojection.png` | 三角化点反投影到图像平面的验证图 |
| `3d_reconstruction.png` | 三维点云 + 拟合圆弧 + 抓取中点的 3D 可视化 |

### needle_3d.npz 字段说明

用 `numpy.load('needle_3d.npz')` 读取：

| 字段 | 形状 | 含义 |
|---|---|---|
| `midpoint_mm` | `(3,)` | 3D 抓取目标点坐标，左相机坐标系，单位 mm |
| `plane_normal` | `(3,)` | 针所在圆弧平面的法向量（单位向量） |
| `radius_mm` | 标量 | 拟合圆弧半径，单位 mm |
| `arc_sweep_deg` | 标量 | 圆弧跨度角度，单位 ° |
| `rms_mm` | 标量 | RANSAC 圆弧拟合残差（RMS），单位 mm |
| `n_inliers` | 标量 | RANSAC 内点数 |
| `n_total` | 标量 | 三角化得到的总 3D 点数 |

---

## 环境配置

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **PyTorch GPU 版本**：`requirements.txt` 中的 `torch` / `torchvision` 为通用版本。若有 NVIDIA GPU，建议替换为对应 CUDA 版本，例如 CUDA 11.8：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> 可在 [pytorch.org](https://pytorch.org/get-started/locally/) 选择合适的版本。

### 3. 安装 SAM3

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

详细说明参见官方仓库：https://github.com/facebookresearch/sam3

### 4. HuggingFace 登录（首次运行时下载模型权重）

```bash
huggingface-cli login
```

---

## 运行 ./script/run_needle_3d.sh

### 前提条件

- 训练好的 YOLOv8 权重文件（位置：yolo_weights/best.pt）
- 左右双目图像各一张（位置：`example_img/left/` 和 `example_img/right/`）
- 相机参数使用的是转换成适配1920*1080图片的 camera_calibration_1920_1080.yaml

### 基本用法

编辑 `script/run_needle_3d.sh` 顶部的默认路径，或通过环境变量/命令行参数传入：

```bash
# 使用脚本内默认路径（example_img/ 下的示例图 + yolo_weights/best.pt）
./script/run_needle_3d.sh

# 指定图像和权重路径
./script/run_needle_3d.sh \
    --left  example_img/left/img_0001.jpg \
    --right example_img/right/img_0001.jpg \
    --weights yolo_weights/best.pt

# 不保存可视化图（仅输出 .npz）
./script/run_needle_3d.sh --no-vis
```

也可通过环境变量设置默认路径：

```bash
export LEFT_IMG=example_img/left/img_0001.jpg
export RIGHT_IMG=example_img/right/img_0001.jpg
export YOLO_WEIGHTS=yolo_weights/best.pt
./script/run_needle_3d.sh
```

### 读取结果

```python
import numpy as np

data = np.load('outputs_3d/needle_3d.npz')
print(data['midpoint_mm'])    # 3D 抓取点 [x, y, z]，单位 mm
print(data['radius_mm'])      # 针的圆弧半径
```

