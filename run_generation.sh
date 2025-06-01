#!/bin/bash
# 文本生成脚本示例

echo "使用训练好的模型生成文本..."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查检查点是否存在
if [ ! -f "./checkpoints/best.pt" ]; then
    echo "错误: 找不到最佳模型检查点 ./checkpoints/best.pt"
    echo "请先运行训练脚本"
    exit 1
fi

if [ ! -f "./checkpoints/config.json" ]; then
    echo "错误: 找不到配置文件 ./checkpoints/config.json"
    echo "请先运行训练脚本"
    exit 1
fi

# 交互式生成
python generate.py \
    --checkpoint "./checkpoints/best.pt" \
    --config "./checkpoints/config.json" \
    --device auto \
    --interactive

echo "生成完成！"
