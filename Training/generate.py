#!/usr/bin/env python3
"""
模型推理和文本生成脚本
"""
import argparse
import json
import torch
from pathlib import Path

from Architecture.model import BasicsTransformerLM


class SimpleTokenizer:
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
        
        print(f"构建词汇表完成，大小: {len(self.char_to_id)}")
        
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


def load_model_and_tokenizer(checkpoint_path: str, config_path: str, device: str):
    """加载模型和分词器"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建模型
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta']
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载完成，步骤: {checkpoint['step']}")
    print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # 创建分词器（需要重新构建词汇表）
    # 注意：在实际应用中，应该保存分词器的词汇表
    tokenizer = SimpleTokenizer(config['vocab_size'])
    
    return model, tokenizer, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cpu"
):
    """生成文本"""
    model.eval()
    
    # 编码提示
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"提示: '{prompt}'")
    print(f"编码后的输入长度: {len(input_ids)}")
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # 解码生成的token
    generated_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def interactive_generation(model, tokenizer, config, device):
    """交互式文本生成"""
    print("\n" + "="*50)
    print("交互式文本生成")
    print("输入 'quit' 退出")
    print("="*50)
    
    while True:
        try:
            prompt = input("\n请输入提示文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出程序")
                break
            
            if not prompt:
                print("请输入有效的提示文本")
                continue
            
            # 生成参数
            max_tokens = input("最大生成token数 (默认100): ").strip()
            max_tokens = int(max_tokens) if max_tokens else 100
            
            temperature = input("温度 (默认1.0): ").strip()
            temperature = float(temperature) if temperature else 1.0
            
            top_k = input("Top-k (默认None): ").strip()
            top_k = int(top_k) if top_k else None
            
            print("\n生成中...")
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            print(f"\n生成结果:")
            print(f"'{generated_text}'")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断")
            break
        except Exception as e:
            print(f"生成过程中出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="模型推理和文本生成")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--config", type=str, required=True,
                       help="模型配置文件路径")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"], help="设备")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, default=None,
                       help="生成提示（如果提供则运行单次生成）")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="生成温度")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k采样")
    
    # 交互模式
    parser.add_argument("--interactive", action="store_true",
                       help="交互式生成模式")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
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
    
    print(f"使用设备: {device}")
    
    # 加载模型
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint, args.config, device
    )
    
    # 需要重新构建词汇表（这是简化版本，实际应用中应该保存词汇表）
    print("警告: 需要重新构建词汇表，请确保使用相同的训练数据")
    train_data_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    if Path(train_data_path).exists():
        print("从训练数据重建词汇表...")
        texts = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000:  # 只用前10000行
                    break
                texts.append(line.strip())
        tokenizer.build_vocab(texts)
    else:
        print("警告: 无法找到训练数据，使用默认词汇表")
        # 创建一个最小词汇表
        default_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\n")
        tokenizer.build_vocab(default_chars)
    
    if args.interactive:
        # 交互式模式
        interactive_generation(model, tokenizer, config, device)
    else:
        # 单次生成模式
        if args.prompt is None:
            args.prompt = "Once upon a time"
        
        print(f"使用提示: '{args.prompt}'")
        generated_text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print(f"\n生成结果:")
        print(f"'{generated_text}'")


if __name__ == "__main__":
    main()
