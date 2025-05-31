import numpy
import numpy.typing as npt
import numpy as np
import torch
import os
from typing import Optional, Dict

def load_dataset_mmap(file_path: str, dtype: np.dtype = np.int32) -> npt.NDArray:
    return np.memmap(file_path, dtype=dtype, mode='r')

def load_text_dataset(
    txt_file_path: str, 
    binary_cache_path: Optional[str] = None,
    vocab_size: Optional[int] = None,
    dtype: np.dtype = np.int32
) -> tuple[npt.NDArray, Dict[str, int], Dict[int, str]]:
    """
    从txt文件加载文本数据集，转换为mmap格式
    
    Args:
        txt_file_path: .txt文件路径
        binary_cache_path: 二进制缓存文件路径，如果为None则自动生成
        vocab_size: 词汇表大小限制，None表示不限制
        dtype: 数据类型，默认为np.int32
        
    Returns:
        (dataset, char_to_int, int_to_char): mmap数组和字符映射
    """
    # 生成缓存文件路径
    if binary_cache_path is None:
        base_name = os.path.splitext(txt_file_path)[0]
        binary_cache_path = f"{base_name}_cache.bin"
    
    # 检查是否存在缓存文件
    cache_exists = os.path.exists(binary_cache_path)
    vocab_cache_path = f"{binary_cache_path}.vocab"
    vocab_exists = os.path.exists(vocab_cache_path)
    
    if cache_exists and vocab_exists:
        # 加载缓存的词汇表
        vocab_data = np.load(vocab_cache_path, allow_pickle=True).item()
        char_to_int = vocab_data['char_to_int']
        int_to_char = vocab_data['int_to_char']
        
        # 加载mmap数据
        dataset = load_dataset_mmap(binary_cache_path, dtype)
        print(f"从缓存加载数据集: {len(dataset)} tokens")
        
    else:
        # 读取文本文件
        print(f"读取文本文件: {txt_file_path}")
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 创建字符到整数的映射
        unique_chars = sorted(list(set(text)))
        if vocab_size and len(unique_chars) > vocab_size:
            # 如果指定了词汇表大小限制，保留最常见的字符
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # 按频率排序，保留前vocab_size个字符
            sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
            unique_chars = [char for char, _ in sorted_chars[:vocab_size]]
            
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_to_char = {i: char for i, char in enumerate(unique_chars)}
        
        print(f"词汇表大小: {len(unique_chars)}")
        print(f"文本长度: {len(text)} 字符")
        
        # 将文本转换为整数序列
        text_ints = []
        unknown_char_id = len(unique_chars) - 1  # 使用最后一个作为未知字符
        
        for char in text:
            if char in char_to_int:
                text_ints.append(char_to_int[char])
            else:
                text_ints.append(unknown_char_id)  # 未知字符
        
        # 保存为二进制文件
        create_binary_dataset(text_ints, binary_cache_path, dtype)
        
        # 保存词汇表
        vocab_data = {
            'char_to_int': char_to_int,
            'int_to_char': int_to_char
        }
        np.save(vocab_cache_path, vocab_data)
        
        # 加载mmap数据
        dataset = load_dataset_mmap(binary_cache_path, dtype)
        print(f"创建数据集缓存: {len(dataset)} tokens")
    
    return dataset, char_to_int, int_to_char

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_len = dataset.shape[0]  # 修复：将len改为dataset_len，避免覆盖内置函数
    
    # 添加边界检查
    if dataset_len <= context_length:
        raise ValueError(f"Dataset length ({dataset_len}) must be greater than context_length ({context_length})")
    
    ret_input = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    ret_target = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # 修复：确保索引不会越界
        max_idx = dataset_len - context_length - 1
        idx = np.random.randint(0, max_idx)
        ret_input[i,:] = torch.tensor(dataset[idx : idx + context_length], device=device)
        ret_target[i,:] = torch.tensor(dataset[idx + 1 : idx + context_length + 1], device=device)
    
    return ret_input, ret_target

def create_binary_dataset(text_data: list[int], output_path: str, dtype: np.dtype = np.int32) -> None:
    """
    将文本数据保存为二进制文件，用于mmap加载
    
    Args:
        text_data: 整数列表形式的文本数据
        output_path: 输出文件路径
        dtype: 数据类型，默认为np.int32
    """
    arr = np.array(text_data, dtype=dtype)
    arr.tofile(output_path)

def decode_text(token_ids: list[int], int_to_char: Dict[int, str]) -> str:
    """
    将token ids解码回文本
    
    Args:
        token_ids: token id列表
        int_to_char: 整数到字符的映射
        
    Returns:
        解码后的文本
    """
    return ''.join([int_to_char.get(token_id, '<UNK>') for token_id in token_ids])

# 使用示例：
def example_usage():
    """
    使用示例：从txt文件加载数据集
    """
    # 1. 从txt文件加载数据集
    txt_file = "sample.txt"  # 替换为你的txt文件路径
    
    # 创建示例文件（如果不存在）
    if not os.path.exists(txt_file):
        sample_text = "Hello world! This is a sample text for testing the dataloader. " * 100
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    # 加载数据集
    dataset, char_to_int, int_to_char = load_text_dataset(txt_file)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"词汇表大小: {len(char_to_int)}")
    print(f"前10个字符: {list(char_to_int.keys())[:10]}")
    
    # 2. 获取批次数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_input, batch_target = get_batch(dataset, batch_size=4, context_length=32, device=device)
    
    print(f"Input shape: {batch_input.shape}")
    print(f"Target shape: {batch_target.shape}")
    
    # 3. 解码示例
    sample_input = batch_input[0].cpu().numpy().tolist()
    sample_target = batch_target[0].cpu().numpy().tolist()
    
    print(f"输入文本: '{decode_text(sample_input, int_to_char)}'")
    print(f"目标文本: '{decode_text(sample_target, int_to_char)}'")

if __name__ == "__main__":
    example_usage()
