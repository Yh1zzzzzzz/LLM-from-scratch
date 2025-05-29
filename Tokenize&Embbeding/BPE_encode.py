import os
from typing import BinaryIO
import io
import regex
import heapq
import multiprocessing as mp
from collections import defaultdict
""""
This is juSt a basic mudole for BPE encoding.

I will finish a Complete BPE class(PreTokenize + encode + decode) in the next file.

"""
vocab = {}  #dict[int ,bytes]  # 字典，存储字节对及其对应的索引
merges = []  #list[tuple[bytes, bytes]]  # 列表，存储合并的字节对
special_tokens = list()  # list[str]  # 特殊标记列表
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
cpu_count = mp.cpu_count()


def pre_tokenize(text: str) -> list[str]:
    return regex.findall(PAT, text)
def pre_token_iterate(text: str):
    return regex.finditer(PAT, text)

    
def encode(text: str) -> list[int]:
    """
    Encode the input text using BPE encoding.
    """
    # Pre-tokenize the text
    #step1 : regex : 
    Seperated_word = pre_tokenize(text)
    special_tokens_byte = [token.encode('utf-8') for token in special_tokens]  # Convert special tokens to bytes
    
    # 标记哪些是特殊token
    tokens = []
    is_special = []
    
    # Convert the list of strings to a list of bytes
    for word in Seperated_word:
        word_bytes = word.encode('utf-8')  # Convert each word to bytes
        if word_bytes in special_tokens_byte:
            tokens.append(word_bytes)
            is_special.append(True)
        else:
            # 转换为bytes对象列表，便于后续合并
            tokens.append([bytes([b]) for b in word_bytes])
            is_special.append(False)
    
    #step 2 : merge
    # 按merges顺序依次应用每个合并规则
    for merge_pair in merges:
        for index in range(len(tokens)):
            if is_special[index]:  # 跳过特殊token
                continue
            
            # 在当前token中查找并合并所有的merge_pair
            i = 0
            while i < len(tokens[index]) - 1:
                # 确保比较的是bytes对象
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
    
    # step3 : convert to int
    reverse_vocab = {v: k for k, v in vocab.items()}  # Reverse the vocab dictionary
    encoded_list = []
    
    for idx, token in enumerate(tokens):
        if is_special[idx]:  # 特殊token直接转换
            if token in reverse_vocab:
                encoded_list.append(reverse_vocab[token])
        else:  # 普通token逐个转换
            for sub_token in token:
                if sub_token in reverse_vocab:
                    encoded_list.append(reverse_vocab[sub_token])
        
    return encoded_list