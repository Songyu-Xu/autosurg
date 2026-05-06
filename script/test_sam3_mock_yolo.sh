#!/bin/bash

# 验证 box 位置是否正确（不需要 SAM3）
python test_sam3_mock_yolo.py \
    --image /home/songyu/Datasets/examples/suturing_needle/images/train/00138.png \
    --label /home/songyu/Datasets/examples/suturing_needle/labels/train/00138.txt \
    --no_sam3

# 跑完整SAM3分割
# python test_sam3_mock_yolo.py \
#     --image /home/songyu/Datasets/examples/suturing_needle/images/train/00138.png \
#     --label /home/songyu/Datasets/examples/suturing_needle/labels/train/00138.txt