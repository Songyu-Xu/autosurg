#!/usr/bin/env bash
# Run stereo 3D needle reconstruction.
# Usage:
#   ./run_needle_3d.sh                        # use defaults below
#   ./run_needle_3d.sh --left path/to/L.jpg --right path/to/R.jpg
#   ./run_needle_3d.sh --calib stereo_calib.npz --no-vis
# All extra arguments are forwarded to needle_3d_reconstruction.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── GPU selection (set CUDA_VISIBLE_DEVICES to override, e.g. GPU=1 ./run_needle_3d.sh) ──
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ── Default paths (edit to match your setup) ─────────────────────────────────

# test_black_background
# LEFT_IMG="${LEFT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_black_background/left/img_0001.jpg}"
# RIGHT_IMG="${RIGHT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_black_background/right/img_0001.jpg}"
# YOLO_WEIGHTS="${YOLO_WEIGHTS:-/home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle_finetune/cuhk_v1/weights/best.pt}"
# OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_3d/May13/test_black_backgound/cropped_left_skeleton_only/img_0001}"

# test_peg_paddle
# LEFT_IMG="${LEFT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle/left/img_0001.jpg}"
# RIGHT_IMG="${RIGHT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle/right/img_0001.jpg}"
# YOLO_WEIGHTS="${YOLO_WEIGHTS:-/home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle_finetune/cuhk_v1/weights/best.pt}"
# OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_3d/May13/test_peg_paddle/img_0001/}"

# test_peg_paddle_2
# LEFT_IMG="${LEFT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle_2/left/img_0001.jpg}"
# RIGHT_IMG="${RIGHT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle_2/right/img_0001.jpg}"
# YOLO_WEIGHTS="${YOLO_WEIGHTS:-/home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle_finetune/cuhk_v1/weights/best.pt}"
# OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_3d/May13/test_peg_paddle_2/img_0001/}"

# test_peg_paddle_3
LEFT_IMG="${LEFT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle_3/left/img_0001.jpg}"
RIGHT_IMG="${RIGHT_IMG:-/home/songyu/Datasets/Autosurg/cuhk/May13_test/Test_image/test_peg_paddle_3/right/img_0001.jpg}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-/home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle_finetune/cuhk_v1/weights/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_3d/May13/test_peg_paddle_3/img_0001_text_prompt/}"

# Optional: point to a stereo_calib.npz to override the built-in calibration.
# CALIB_NPZ="${SCRIPT_DIR}/stereo_calib.npz"

# ── Activate conda / venv if needed ──────────────────────────────────────────
# Uncomment and adjust one of the lines below if you use an environment.
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate autosurg
source "${PROJECT_DIR}/.venv/bin/activate"

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --left        "${LEFT_IMG}"
    --right       "${RIGHT_IMG}"
    --weights     "${YOLO_WEIGHTS}"
    --output-dir  "${OUTPUT_DIR}"
)

if [[ -n "${CALIB_NPZ:-}" && -f "${CALIB_NPZ}" ]]; then
    ARGS+=(--calib "${CALIB_NPZ}")
fi

# Append any extra CLI arguments passed to this script.
ARGS+=("$@")

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Stereo 3D Needle Reconstruction"
echo "============================================================"
echo "  left    : ${LEFT_IMG}"
echo "  right   : ${RIGHT_IMG}"
echo "  weights : ${YOLO_WEIGHTS}"
echo "  output  : ${OUTPUT_DIR}"
echo "  GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "------------------------------------------------------------"

python "${PROJECT_DIR}/needle_3d_reconstruction.py" "${ARGS[@]}"


echo "------------------------------------------------------------"
echo "Done. Results saved to: ${OUTPUT_DIR}"
