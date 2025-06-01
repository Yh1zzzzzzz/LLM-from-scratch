#!/usr/bin/env python3
"""
完整的Transformer模型训练脚本
支持从TinyStories数据集训练语言模型
"""
import os
import json
import argparse
import logging
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from Architecture.model import BasicsTransformerLM
from Optimizer.AdamW import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TinyStoriesDataset(Dataset):
    """TinyStories数据集类"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 读取数据
        logger.info(f"加载数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        
        # 过滤空行
        self.texts = [text.strip() for text in self.texts if text.strip()]
        logger.info(f"数据集大小: {len(self.texts)} 条记录")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充到指定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 转换为tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # 对于语言建模，输入和目标相同但错位一位
        if len(input_ids) < self.max_length:
            # 如果序列太短，使用padding
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
        
        # 输入和目标
        x = input_ids[:-1]  # 输入序列
        y = input_ids[1:]   # 目标序列（向右移动一位）
        
        return x, y


class SimpleTokenizer:
    """简单的字符级分词器（备用方案）"""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_built = False
        
    def build_vocab(self, texts):
        """从文本中构建词汇表"""
        char_counts = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # 按频率排序，保留最常见的字符
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 保留特殊token的位置
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        vocab_chars = special_tokens.copy()
        
        # 添加最常见的字符
        for char, _ in sorted_chars:
            if len(vocab_chars) >= self.vocab_size:
                break
            if char not in vocab_chars:
                vocab_chars.append(char)
        
        # 构建映射
        self.char_to_id = {char: i for i, char in enumerate(vocab_chars)}
        self.id_to_char = {i: char for i, char in enumerate(vocab_chars)}
        self.vocab_built = True
        
        logger.info(f"构建词汇表完成，大小: {len(self.char_to_id)}")
        
    def encode(self, text):
        """编码文本为token ids"""
        if not self.vocab_built:
            raise RuntimeError("词汇表未构建，请先调用build_vocab")
        
        # 添加起始token
        tokens = [self.char_to_id['<bos>']]
        
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id['<unk>'])
        
        # 添加结束token
        tokens.append(self.char_to_id['<eos>'])
        
        return tokens
    
    def decode(self, token_ids):
        """解码token ids为文本"""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                if char not in ['<pad>', '<unk>', '<bos>', '<eos>']:
                    chars.append(char)
        return ''.join(chars)


class Trainer:
    """训练器类"""
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 2000,
        gradient_clip_norm: float = 1.0
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.gradient_clip_norm = gradient_clip_norm
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
        
        # 训练状态
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def train_step(self, batch):
        """单个训练步骤"""
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # 前向传播
        logits = self.model(x)
        
        # 计算损失
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        for batch in self.val_dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新检查点
        latest_path = self.save_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，也保存一份
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        # 定期保存带步数的检查点
        if self.step % (self.save_interval * 5) == 0:
            step_path = self.save_dir / f'step_{self.step}.pt'
            torch.save(checkpoint, step_path)
    
    def train(self, num_epochs: int):
        """训练循环"""
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"总训练步数: {total_steps}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # 训练一步
                loss = self.train_step(batch)
                self.step += 1
                
                # 记录日志
                if self.step % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.step / elapsed
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {self.step}/{total_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"Steps/sec: {steps_per_sec:.2f}"
                    )
                
                # 评估
                if self.step % self.eval_interval == 0:
                    val_loss, perplexity = self.evaluate()
                    logger.info(f"验证 - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    # 保存最佳模型
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    self.save_checkpoint(is_best=is_best)
                
                # 定期保存
                elif self.step % self.save_interval == 0:
                    self.save_checkpoint()
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} 完成，耗时: {epoch_time:.2f}秒")
        
        # 训练结束，最终评估
        final_val_loss, final_perplexity = self.evaluate()
        total_time = time.time() - start_time
        
        logger.info(f"训练完成!")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"最终验证损失: {final_val_loss:.4f}")
        logger.info(f"最终困惑度: {final_perplexity:.2f}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0
):
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = TinyStoriesDataset(train_path, tokenizer, max_length)
    val_dataset = TinyStoriesDataset(val_path, tokenizer, max_length)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataloader, val_dataloader


def create_model(config: dict, device: str) -> BasicsTransformerLM:
    """创建模型"""
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta']
    )
    
    model = model.to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    
    return model


def build_simple_tokenizer(train_path: str, vocab_size: int = 1000):
    """构建简单分词器"""
    logger.info("构建简单字符级分词器...")
    
    # 读取训练数据的一部分来构建词汇表
    texts = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # 只用前10000行构建词汇表
                break
            texts.append(line.strip())
    
    tokenizer = SimpleTokenizer(vocab_size)
    tokenizer.build_vocab(texts)
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练Transformer语言模型")
    
    # 数据参数
    parser.add_argument("--train-data", type=str, 
                       default="../../data/TinyStoriesV2-GPT4-train.txt",
                       help="训练数据路径")
    parser.add_argument("--val-data", type=str,
                       default="../../data/TinyStoriesV2-GPT4-valid.txt", 
                       help="验证数据路径")
    
    # 模型参数
    parser.add_argument("--vocab-size", type=int, default=1000, help="词汇表大小")
    parser.add_argument("--context-length", type=int, default=512, help="上下文长度")
    parser.add_argument("--d-model", type=int, default=256, help="模型维度")
    parser.add_argument("--num-layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=1024, help="前馈网络维度")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta参数")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--num-epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, help="梯度裁剪范数")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "mps", "cpu"], help="设备")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="保存目录")
    parser.add_argument("--log-interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--eval-interval", type=int, default=1000, help="评估间隔")
    parser.add_argument("--save-interval", type=int, default=2000, help="保存间隔")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载器工作进程数")
    parser.add_argument("--max-length", type=int, default=256, help="最大序列长度")
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"训练数据文件不存在: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"验证数据文件不存在: {args.val_data}")
    
    # 设备设置
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    
    # 构建分词器
    tokenizer = build_simple_tokenizer(args.train_data, args.vocab_size)
    
    # 创建数据加载器
    train_dataloader, val_dataloader = create_dataloaders(
        args.train_data,
        args.val_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # 模型配置
    model_config = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta
    }
    
    # 创建模型
    model = create_model(model_config, device)
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 保存配置
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"配置已保存到: {config_path}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        gradient_clip_norm=args.gradient_clip_norm
    )
    
    # 开始训练
    trainer.train(args.num_epochs)
    
    logger.info("训练脚本执行完成！")


if __name__ == "__main__":
    main()
