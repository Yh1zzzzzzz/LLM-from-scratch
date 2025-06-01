# Transformer模型训练和推理

本项目提供了完整的Transformer语言模型训练和推理脚本，基于TinyStories数据集进行训练。

## 项目结构

```
LLM_scratch/
├── Architecture/          # 模型架构组件
│   ├── model.py           # 主模型文件
│   ├── FFN.py            # 前馈网络
│   ├── MultiHeadAtten.py # 多头注意力
│   ├── RMSNorm.py        # RMS归一化
│   └── nn_utils.py       # 工具函数
├── Optimizer/             # 优化器
│   └── AdamW.py          # AdamW优化器
├── train.py              # 训练脚本
├── generate.py           # 生成脚本
├── run_training.sh       # 训练运行脚本
├── run_generation.sh     # 生成运行脚本
└── checkpoints/          # 模型检查点目录
```

## 快速开始

### 1. 数据准备

确保TinyStories数据集位于正确路径：
```
../../cs336/hw/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt
../../cs336/hw/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt
```

### 2. 训练模型

运行训练脚本：
```bash
./run_training.sh
```

或者手动运行：
```bash
python train.py \
    --train-data "../../cs336/hw/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt" \
    --val-data "../../cs336/hw/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt" \
    --vocab-size 1000 \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --batch-size 8 \
    --num-epochs 5 \
    --learning-rate 1e-4
```

### 3. 文本生成

训练完成后，使用生成脚本：
```bash
./run_generation.sh
```

或者手动运行：
```bash
python generate.py \
    --checkpoint "./checkpoints/best.pt" \
    --config "./checkpoints/config.json" \
    --interactive
```

## 训练参数说明

### 模型参数
- `--vocab-size`: 词汇表大小 (默认: 1000)
- `--context-length`: 上下文长度 (默认: 512)
- `--d-model`: 模型维度 (默认: 256)
- `--num-layers`: Transformer层数 (默认: 6)
- `--num-heads`: 注意力头数 (默认: 8)
- `--d-ff`: 前馈网络维度 (默认: 1024)

### 训练参数
- `--batch-size`: 批次大小 (默认: 8)
- `--num-epochs`: 训练轮数 (默认: 10)
- `--learning-rate`: 学习率 (默认: 1e-4)
- `--weight-decay`: 权重衰减 (默认: 0.01)
- `--gradient-clip-norm`: 梯度裁剪范数 (默认: 1.0)

### 其他参数
- `--device`: 设备选择 (auto/cuda/mps/cpu)
- `--save-dir`: 检查点保存目录 (默认: ./checkpoints)
- `--log-interval`: 日志间隔 (默认: 100)
- `--eval-interval`: 评估间隔 (默认: 1000)

## 生成参数说明

- `--prompt`: 生成提示文本
- `--max-tokens`: 最大生成token数 (默认: 100)
- `--temperature`: 生成温度 (默认: 1.0)
- `--top-k`: Top-k采样 (默认: None)
- `--interactive`: 交互式生成模式

## 示例用法

### 1. 小规模快速训练
```bash
python train.py \
    --vocab-size 500 \
    --d-model 128 \
    --num-layers 4 \
    --num-heads 4 \
    --batch-size 4 \
    --num-epochs 2 \
    --max-length 128
```

### 2. 单次文本生成
```bash
python generate.py \
    --checkpoint "./checkpoints/best.pt" \
    --config "./checkpoints/config.json" \
    --prompt "Once upon a time" \
    --max-tokens 50 \
    --temperature 0.8
```

### 3. 交互式生成
```bash
python generate.py \
    --checkpoint "./checkpoints/best.pt" \
    --config "./checkpoints/config.json" \
    --interactive
```

## 注意事项

1. **内存使用**: 根据你的硬件调整批次大小和序列长度
2. **设备选择**: 脚本会自动检测可用设备（CUDA > MPS > CPU）
3. **词汇表**: 使用简单的字符级分词器，适合小规模实验
4. **检查点**: 训练过程中会自动保存最佳模型和定期检查点
5. **日志**: 训练日志会显示损失、困惑度和吞吐量信息

## 监控训练

训练过程中会输出：
- 每步损失值
- 验证损失和困惑度
- 训练速度（steps/sec）
- 模型参数数量

## 故障排除

1. **CUDA内存不足**: 减少批次大小或序列长度
2. **数据文件不存在**: 检查数据路径是否正确
3. **权限问题**: 确保脚本有执行权限 (`chmod +x *.sh`)
4. **Python路径**: 确保正确设置了PYTHONPATH

## 扩展功能

脚本支持以下扩展：
- 添加更复杂的分词器（BPE等）
- 实现学习率调度
- 添加更多评估指标
- 支持分布式训练
- 模型量化和优化

## 性能优化建议

1. 使用适当的批次大小充分利用GPU
2. 启用混合精度训练（需要修改代码）
3. 使用数据并行加速训练
4. 优化数据加载流水线
