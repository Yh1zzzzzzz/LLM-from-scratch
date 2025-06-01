#!/bin/bash
# 训练脚本示例

echo "开始训练Transformer模型..."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 基本训练配置
python train.py \
    --train-data "../../data/TinyStoriesV2-GPT4-valid.txt" \
    --val-data "../../data/TinyStoriesV2-GPT4-valid.txt" \
    --vocab-size 1000 \
    --context-length 512 \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --d-ff 1024 \
    --batch-size 8 \
    --num-epochs 5 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --gradient-clip-norm 1.0 \
    --device auto \
    --save-dir "./checkpoints" \
    --log-interval 50 \
    --eval-interval 500 \
    --save-interval 1000 \
    --max-length 256

echo "训练完成！"
