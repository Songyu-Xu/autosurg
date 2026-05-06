# Surgical Needle Detection & Segmentation Pipeline

An end-to-end automated pipeline for detecting and segmenting surgical needles in images, combining **YOLOv8** for object detection and **SAM3** for precise instance segmentation.

---

## How It Works

The pipeline runs in three sequential steps:

```
Input Image → [Step 1] YOLOv8 Detection + Box Enlargement
            → [Step 2] SAM3 Fine Segmentation
            → [Step 3] Visualization & Save
```

**Step 1 — YOLOv8 Detection + Box Enlargement**
A trained YOLOv8 model locates surgical needles in the image and returns bounding boxes. Each box is then expanded by 40% (configurable) around its center to provide sufficient context for the segmentation model.

**Step 2 — SAM3 Segmentation**
Each enlarged bounding box is passed to SAM3 (Segment Anything Model 3) along with a text prompt (`"suturing needle"`) for more stable results on small objects. SAM3 outputs a pixel-level mask for each detected needle, and the highest-confidence mask is selected.

**Step 3 — Visualization & Save**
The masks are overlaid on the original image as semi-transparent color fills, and the bounding boxes are drawn with confidence scores. The result is saved as a new image file.

---

## Installation

### 1. Create a Virtual Environment

It is strongly recommended to install all dependencies inside a Python virtual environment to avoid conflicts with other projects.

```bash
# Navigate to your project directory
cd autosurg

# Create the virtual environment
python -m venv .venv

# Activate it — macOS / Linux
source .venv/bin/activate

# Activate it — Windows
.venv\Scripts\activate
```

Once activated, your terminal prompt will show a `(.venv)` prefix. All packages installed after this point are isolated to this environment.

> To deactivate the environment at any time, simply run `deactivate`.

### 2. Install Core Dependencies

```bash
pip install ultralytics numpy opencv-python Pillow
```

### 3. Install PyTorch

SAM3 is a large model and will run very slowly on CPU. Installing the GPU-enabled version of PyTorch is strongly recommended if you have an NVIDIA GPU.

```bash
# CPU only
pip install torch torchvision

# NVIDIA GPU — choose the build matching your CUDA version, e.g. CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If a CUDA-capable GPU is available, it will be used automatically. Otherwise the pipeline falls back to CPU.

### 4. Install SAM3 (from Source)

SAM3 must be installed from its GitHub repository. The `-e` flag installs it in editable mode, linking the source code directly into the virtual environment.

```bash
# Clone into the project directory (or any preferred location)
git clone https://github.com/facebookresearch/sam3
cd sam3
pip install -e .
cd ..
```

### 5. HuggingFace Login

SAM3 model weights are downloaded automatically from HuggingFace on the first run. A free HuggingFace account is required:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Prerequisites

You need a **trained YOLOv8 weight file** (`.pt`) for surgical needle detection. The default path is:

```
runs/needle/exp1/weights/best.pt
```

You can override this with the `--yolo_weights` argument (see Usage below).

---

## Usage

### Single Image

```bash
python detection_to_segmentation.py --image path/to/image.jpg
```

### Single Image with Custom Weights

```bash
python detection_to_segmentation.py --image path/to/image.jpg --yolo_weights path/to/best.pt
```

### Batch Processing (Entire Folder)

```bash
python detection_to_segmentation.py --image_dir path/to/folder/ --yolo_weights path/to/best.pt
```

Supported image formats for batch mode: `.jpg`, `.jpeg`, `.png`, `.bmp`

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--image` | — | Path to a single input image |
| `--image_dir` | — | Path to a folder for batch processing |
| `--yolo_weights` | `runs/needle/exp1/weights/best.pt` | Path to YOLOv8 weight file |
| `--conf` | `0.25` | YOLO detection confidence threshold |
| `--expand` | `1.4` | Bounding box expansion ratio (1.4 = 40% enlargement) |

---

## Output

For each processed image, a result file named `<original_name>_result.jpg` is saved in the same directory. The result image contains:

- Detected needles highlighted with color-coded semi-transparent mask overlays
- Enlarged bounding boxes with confidence scores

The pipeline also returns a Python dictionary for downstream use (e.g., robotic grasping):

```python
{
    "boxes_xyxy": np.ndarray,   # Enlarged bounding boxes [N, 4]
    "masks":      list,          # Per-needle binary masks [H, W] bool
    "scores":     list,          # SAM3 confidence scores per mask
}
```

---

## Configuration

Key parameters can be adjusted at the top of the script:

```python
YOLO_WEIGHTS = "runs/needle/exp1/weights/best.pt"  # Path to trained weights
YOLO_CONF    = 0.25       # Detection confidence threshold
BOX_EXPAND   = 1.4        # Bounding box enlargement factor
SAM3_TEXT    = "suturing needle"  # Text prompt passed to SAM3
```

---

## Dependencies Summary

| Package | Purpose | Install |
|---|---|---|
| `ultralytics` | YOLOv8 detection | `pip install ultralytics` |
| `sam3` | Instance segmentation | Install from source (see above) |
| `torch` | Deep learning backend | `pip install torch` |
| `opencv-python` | Image I/O and visualization | `pip install opencv-python` |
| `Pillow` | Image loading for SAM3 | `pip install Pillow` |
| `numpy` | Array operations | `pip install numpy` |
| `huggingface_hub` | Downloading SAM3 weights | `pip install huggingface_hub` |