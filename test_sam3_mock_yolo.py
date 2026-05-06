"""
模拟 YOLO 输出 → SAM3 分割 测试脚本
用途：在 YOLO 尚未训练完成时，用标注 txt 文件模拟检测结果，测试 SAM3 效果

用法：
    python test_sam3_with_mock_yolo.py --image 00139.png --label 00138.txt
    python test_sam3_with_mock_yolo.py --image 00139.png --label 00138.txt --no_sam3  # 仅验证 box，不跑 SAM3
"""

import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EXPAND   = 1.4          # box 放大比例
SAM3_TEXT = "suturing needle"


# ───────── Step 1: 从 YOLO 格式 txt 读取 box ─────────
def load_boxes_from_yolo_txt(label_path: str, img_w: int, img_h: int,
                              expand: float = EXPAND):
    """
    读取 YOLO 格式标注文件，返回放大后的 xyxy 绝对坐标。
    YOLO 格式：class cx cy w h  (均归一化到 [0,1])
    """
    boxes_xyxy = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), \
                               float(parts[3]), float(parts[4])

            # 归一化 → 绝对坐标
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            # 放大
            ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
            nw  = (x2 - x1) * expand
            nh  = (y2 - y1) * expand
            nx1 = max(0,     ncx - nw / 2)
            ny1 = max(0,     ncy - nh / 2)
            nx2 = min(img_w, ncx + nw / 2)
            ny2 = min(img_h, ncy + nh / 2)

            boxes_xyxy.append([nx1, ny1, nx2, ny2])
            print(f"  原始 box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"  放大后:   [{nx1:.1f}, {ny1:.1f}, {nx2:.1f}, {ny2:.1f}]")

    return np.array(boxes_xyxy)


# ───────── Step 2: SAM3 分割（与原 pipeline 完全一致）─────────
def xyxy_to_xywh_norm(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return np.array([cx, cy, w, h], dtype=np.float32)


def sam3_segment(image_path, boxes_xyxy, img_w, img_h, text_prompt=SAM3_TEXT):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # ← 同时开启 tf32（notebook 也有这两行）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[SAM3] 加载模型...")
    model = build_sam3_image_model().to(DEVICE)
    processor = Sam3Processor(model)
    image = Image.open(image_path).convert("RGB")

    all_masks, all_scores = [], []

    # ← 用 with 语法，更安全，推理结束后自动退出 autocast 上下文
    with torch.autocast("cuda", dtype=torch.bfloat16):  # 开启自动混合精度上下文。自动把计算中涉及的输入 tensor 转换为 bfloat16，使其与模型权重的类型保持一致，从而避免矩阵乘法时的类型冲突报错
        for i, box_xyxy in enumerate(boxes_xyxy):
            print(f"[SAM3] 处理第 {i+1}/{len(boxes_xyxy)} 个目标...")
            state  = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=text_prompt)

            box_norm = xyxy_to_xywh_norm(box_xyxy, img_w, img_h)
            output   = processor.add_geometric_prompt(box=box_norm, label=1, state=state)

            masks  = output["masks"].cpu().float().numpy()
            scores = output["scores"].cpu().float().numpy()

            best_idx   = scores.argmax()
            best_mask  = masks[best_idx, 0].astype(bool)
            best_score = float(scores[best_idx])

            all_masks.append(best_mask)
            all_scores.append(best_score)
            print(f"  → mask score: {best_score:.3f}")

    return all_masks, all_scores


# ───────── Step 3: 可视化 ─────────
def visualize(image_path, boxes_xyxy, masks=None, scores=None, output_path=None):
    img     = cv2.imread(image_path)
    overlay = img.copy()
    color   = (0, 255, 100)

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)

        # mask（有 SAM3 结果时才画）
        if masks is not None:
            overlay[masks[i]] = (
                overlay[masks[i]] * 0.4 + np.array(color) * 0.6
            ).astype(np.uint8)

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"needle {scores[i]:.2f}" if scores else "needle (mock)"
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0) if masks else img

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_result.jpg"

    cv2.imwrite(output_path, result)
    print(f"[✓] 结果已保存：{output_path}")
    return result


# ───────── 主函数 ─────────
def main():
    parser = argparse.ArgumentParser(description="模拟 YOLO → SAM3 测试")
    parser.add_argument("--image",   required=True, help="输入图片路径")
    parser.add_argument("--label",   required=True, help="YOLO 格式 txt 标注路径")
    parser.add_argument("--expand",  type=float, default=EXPAND, help="box 放大比例")
    parser.add_argument("--no_sam3", action="store_true",
                        help="仅验证 box 位置，跳过 SAM3（调试用）")
    args = parser.parse_args()

    image = Image.open(args.image)
    img_w, img_h = image.size

    print(f"\n图像尺寸: {img_w} x {img_h}")
    print("[Step 1] 读取模拟 YOLO 输出（来自标注 txt）...")
    boxes_xyxy = load_boxes_from_yolo_txt(args.label, img_w, img_h, args.expand)

    if len(boxes_xyxy) == 0:
        print("[STOP] 标注文件中没有 box")
        return

    if args.no_sam3:
        print("\n[--no_sam3] 跳过 SAM3，仅保存 box 可视化结果")
        visualize(args.image, boxes_xyxy)
        return

    print("\n[Step 2] SAM3 分割...")
    masks, scores = sam3_segment(args.image, boxes_xyxy, img_w, img_h)

    print("\n[Step 3] 保存结果...")
    visualize(args.image, boxes_xyxy, masks, scores)


if __name__ == "__main__":
    main()