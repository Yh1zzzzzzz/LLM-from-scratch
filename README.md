# 🚀 LLM from Scratch
一个从零开始构建大语言模型(LLM)的完整项目，涵盖从**底层算子实现**到模型训练的全栈技术。
# 🤔为什么创建这个项目？

1. **Build From Scratch is the best way to learn!**
2. **为了捡起作者差点忘得干干净净的pytorch、 cuda、 Triton基础**

## 📖 项目概述

本项目旨在深度理解大语言模型的**每一个组件**，所以除了基本的张量操作以及自动求导外，几乎不使用pytorch提供的现成组件(torch.nn, torch.nn.functional,torch.optim等)。从分词器开始，通过**使用Triton、CUDA手工实现所有关键算子，并兼容torch的自动求导**、优化器、损失函数等关键组件。项目采用多种实现方式（Triton、CUDA、Pytorch），提供性能对比和优化洞察。


## 🎯 项目进展

### ✅ 已完成

**以下的所有组件均为手工完成，不使用Pytorch现成组件**
- [✅] 完成了BPE分词器(PreTokenize Encode Decode  后续考虑用Rust｜CPP 重写一个速度更快的版本)
- [✅] Embeddings (Token-Embedding & Rotary Position Embedding(RoPE))
- [✅] Transformer前向传播所需要的所有核心算子实现 (使用Triton、CUDA、Pytorch实现，并兼容torch.autograd)
- [✅] 反向传播所需要的所有部件(LossFunction Optimizer Gradient clipping  Learning rate scheduling)
- [✅] 完整的训练框架(DataLoader CheckPoint Training Loop)
- [✅] 性能基准测试框架
- [✅] 自动求导支持 (所有使用Triton、CUDA实现的算子都兼容torch.autograd.Function)

### 🚧 进行中
- [ ] 模型并行策略
- [ ] 数据处理流水线
- [ ] 分布式训练
- [ ] 模型对齐技术
- [ ] 推理优化
- [ ] 混合专家(MoE)实现


## 🏗️ 项目架构

```
LLM_scratch/
├── 🧮 Operator/                    # 核心算子实现
├── ⚡ activation_function/          # 激活函数实现
├── 📊 benchmark&profiler/          # 性能测试工具
├── 🎯 Alignment/                   # 模型对齐技术
├── 🏛️ Architecture/                # 模型架构设计
├── 🌐 Data_Crawl/                  # 数据采集与处理
├── 🔀 MoE/                         # 混合专家模型
├── ⚖️ Parallelsim/                 # 并行训练策略
├── 📈 Scaling/                     # 模型扩展规律
└── 💾 Training_Data/               # 训练数据管理
```


## 🚀 快速开始

### 环境要求

# Python环境
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.0
Triton >= 2.0


### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/LLM_scratch.git
cd LLM_scratch

# 安装Python依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install triton
pip install numpy matplotlib seaborn
```




## 🤝 贡献指南

欢迎所有形式的贡献！

### 如何贡献
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- ⚡ 性能优化
- 🧪 测试用例

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。


## 📞 联系方式

- 项目维护者: [young](1805112144[at]qq[dot]com & yanght45[at]qq[dot]com)

