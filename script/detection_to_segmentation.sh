#!/bin/bash
# 自动定位脚本所在目录，确保从正确路径激活环境
# cd "$(dirname "$0")"
# source ../.venv/bin/activate


# ─────────────────────────── 单张图片模式 ───────────────────────────
# YOLO 和 SAM3 各加载一次，处理完即退出
# 如果想跳过弧线拟合(比如只测 mask 质量)，加 --no_arc 即可
# python detection_to_segmentation.py \
#     --image /home/songyu/Datasets/Autosurg/cuhk/Cali_Data_Needle_Image/needle_image/left/img_0001.jpg \
#     --yolo_weights /home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle/synthetic_v1/weights/best.pt \
#     --conf 0.01 \
#     --expand 1 \
#     --output results/cuhk/left/img_0001_arc_result.jpg

# 批量处理同目录下所有图片（YOLO + SAM3 各只加载一次）
python detection_to_segmentation.py \
    --image_dir /home/songyu/Datasets/Autosurg/cuhk/Cali_Data_Needle_Image/needle_image/left/ \
    --yolo_weights /home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle_finetune/cuhk_v1/weights/best.pt \
    --conf 0.4 \
    --expand 1 \
    --no_arc \
    --output results/cuhk/left/


# 下游任务中如何拿到中点
# result = run_pipeline(...)
# arc = result["arc_infos"][0]       # 第一个(唯一)目标的拟合结果
# if arc is not None:
#     mid_xy    = arc["midpoint"]     # (mx, my) — 针中点,像素坐标
#     center_xy = arc["center"]       # (cx, cy) — 圆心(持针器理想旋转轴)
#     radius    = arc["radius"]       # 拟合半径
#     endpoints = arc["endpoints"]    # ((x1,y1),(x2,y2)) — 针尖两端


# ─────────────────────────── 批量处理模式 ───────────────────────────
# YOLO 和 SAM3 在循环外各只加载一次，所有图片共用同一模型实例
# 相比逐张调用 run_pipeline，节省了 N-1 次模型初始化的开销
# python detection_to_segmentation.py \
#     --image_dir /home/songyu/Datasets/Autosurg/autosurg/output_frames/ \
#     --yolo_weights /home/songyu/Project/yolo/YOLOv8_needle/runs/detect/runs/needle/synthetic_v1/weights/best.pt \
#     --conf 0.4 \
#     --expand 1.4 \
#     --output results/autosurg_youtube_demo/
