import os
import itertools
from typing import BinaryIO, Iterable, Iterator
import io
import regex
import heapq
import multiprocessing as mp
from collections import defaultdict
import json

class BPE:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # 创建反向词汇表用于编码
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        # 预分词正则表达式模式
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
    @classmethod
    def from_files(cls, vocab_file: str, merges_file: str, special_tokens: list[str] | None = None):
        """从文件加载vocab和merges"""
        # 加载vocab
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        # 转换vocab格式 - 假设文件中是 {token_str: id} 格式
        vocab = {}
        for token_str, token_id in vocab_dict.items():
            if token_str.startswith('<') and token_str.endswith('>'):
                # 特殊token
                vocab[token_id] = token_str.encode('utf-8')
            else:
                # 普通token，可能是字节或合并后的字节序列
                try:
                    # 尝试解析为字节序列
                    vocab[token_id] = bytes.fromhex(token_str)
                except ValueError:
                    # 如果不是hex格式，直接编码
                    vocab[token_id] = token_str.encode('utf-8')
        
        # 加载merges
        merges = []
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2:
                        left = bytes.fromhex(parts[0]) if parts[0].startswith('\\x') else parts[0].encode('utf-8')
                        right = bytes.fromhex(parts[1]) if parts[1].startswith('\\x') else parts[1].encode('utf-8')
                        merges.append((left, right))
        
        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[str]:
        """预分词"""
        return regex.findall(self.PAT, text)

    def encode(self, text: str) -> list[int]:
        """
        Encode the input text using BPE encoding.
        """
        # Pre-tokenize the text
        separated_words = self.pre_tokenize(text)
        special_tokens_byte = [token.encode('utf-8') for token in self.special_tokens]
        
        # 标记哪些是特殊token
        tokens = []
        is_special = []
        
        # Convert the list of strings to a list of bytes
        for word in separated_words:
            word_bytes = word.encode('utf-8')
            if word_bytes in special_tokens_byte:
                tokens.append(word_bytes)
                is_special.append(True)
            else:
                # 转换为bytes对象列表，便于后续合并
                tokens.append([bytes([b]) for b in word_bytes])
                is_special.append(False)
        
        # step 2: merge
        # 按merges顺序依次应用每个合并规则
        for merge_pair in self.merges:
            for index in range(len(tokens)):
                if is_special[index]:  # 跳过特殊token
                    continue
                
                # 在当前token中查找并合并所有的merge_pair
                i = 0
                while i < len(tokens[index]) - 1:
                    left = tokens[index][i]
                    right = tokens[index][i+1]
                    
                    if (left, right) == merge_pair:
                        # 合并字节对
                        merged_byte = left + right
                        tokens[index][i] = merged_byte
                        del tokens[index][i+1]
                        # 不增加i，因为可能有连续合并
                    else:
                        i += 1
        
        # step 3: convert to int
        encoded_list = []
        
        for idx, token in enumerate(tokens):
            if is_special[idx]:  # 特殊token直接转换
                if token in self.reverse_vocab:
                    encoded_list.append(self.reverse_vocab[token])
            else:  # 普通token逐个转换
                for sub_token in token:
                    if sub_token in self.reverse_vocab:
                        encoded_list.append(self.reverse_vocab[sub_token])
        
        return encoded_list

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to text.
        """
        bytes_list = []
        for token_id in ids:
            if token_id in self.vocab:
                bytes_list.append(self.vocab[token_id])
            else:
                raise ValueError(f"ID {token_id} not found in vocabulary.")
        
        # Convert the list of bytes to a single bytes object
        decoded_bytes = b''.join(bytes_list)
        # Decode the bytes object to a string
        decoded_text = decoded_bytes.decode('utf-8')
        return decoded_text

    def encode_iterate(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings, yielding token IDs one by one.
        """
        for text in iterable:
            encoded_ids = self.encode(text)
            for token_id in encoded_ids:
                yield token_id
    
    @classmethod
    def train_from_corpus(cls, 
                         input_path: str,
                         vocab_size: int, 
                         special_tokens: list[str] | None = None) -> 'BPE':
        """
        从语料库训练BPE模型（基于parallel_pre_tokenize的逻辑）
        """
        if special_tokens is None:
            special_tokens = []
            
        cpu_count = mp.cpu_count()
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file {input_path} does not exist.")

        escaped_tokens = [regex.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)
        
        # 初始化词汇表
        vocab = {}  
        for i in range(256):
            vocab[i] = bytes([i])
        for i, token in enumerate(special_tokens):
            vocab[len(vocab)] = token.encode("utf-8")
        
        merges = []
        
        # 获取文件块边界（简化版本，这里使用串行处理）
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # 读取并统计token
        token_count = defaultdict(int)
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = regex.findall(PAT, line)
                for token in tokens:
                    token_count[token] += 1
        
        # 构建字节对计数
        successive = defaultdict(int)
        for token, count in token_count.items():
            token_bytes = token.encode("utf-8")
            if len(token_bytes) > 1:
                for i in range(len(token_bytes) - 1):
                    byte_pair = (bytes([token_bytes[i]]), bytes([token_bytes[i + 1]]))
                    successive[byte_pair] += count
        
        # BPE合并过程
        while len(vocab) < vocab_size:
            if not successive:
                break
                
            most_common_pair = max(successive, key=successive.get)
            merges.append(most_common_pair)
            
            index = len(vocab)
            vocab[index] = most_common_pair[0] + most_common_pair[1]
            
            del successive[most_common_pair]
            
            # 更新字节对计数（简化版本）
            new_successive = defaultdict(int)
            for (b1, b2), count in successive.items():
                new_successive[(b1, b2)] = count
            successive = new_successive
        
        return cls(vocab, merges, special_tokens)
    
    def save(self, vocab_file: str, merges_file: str):
        """保存vocab和merges到文件"""
        # 保存vocab
        vocab_dict = {}
        for token_id, token_bytes in self.vocab.items():
            try:
                # 尝试解码为字符串
                vocab_dict[token_bytes.decode('utf-8')] = token_id
            except UnicodeDecodeError:
                # 如果无法解码，使用hex表示
                vocab_dict[token_bytes.hex()] = token_id
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        # 保存merges
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("# BPE merges file\n")
            for left, right in self.merges:
                try:
                    left_str = left.decode('utf-8')
                    right_str = right.decode('utf-8')
                    f.write(f"{left_str} {right_str}\n")
                except UnicodeDecodeError:
                    f.write(f"\\x{left.hex()} \\x{right.hex()}\n")