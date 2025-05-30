import os
from typing import BinaryIO
import io
import regex
import heapq
import multiprocessing as mp
from collections import defaultdict
""""
This is juSt a basic mudole for BPE decoding.

I will finish a Complete BPE class(PreTokenize + encode + decode) in the next file.

"""
vocab = {}  #dict[int ,bytes]  # 字典，存储字节对及其对应的索引
merges = []  #list[tuple[bytes, bytes]]  # 列表，存储合并的字节对
special_tokens = list()  # list[str]  # 特殊标记列表
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
cpu_count = mp.cpu_count()

def decode(ids: list[int]) -> str:
    bytes_list = []
    for i in ids:
        if i in vocab:
            bytes_list.append(vocab[i])
        else:
            raise ValueError(f"ID {i} not found in vocabulary.")
    # Convert the list of bytes to a single bytes object
    decoded_bytes = b''.join(bytes_list)
    # Decode the bytes object to a string
    decoded_text = decoded_bytes.decode('utf-8')
    return decoded_text
