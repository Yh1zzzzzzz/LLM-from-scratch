import os
from typing import BinaryIO
import io
import regex
import heapq
import multiprocessing as mp
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(args):
    """处理单个文件块的工作函数"""
    start, end, file_path, pattern = args
    token_count = defaultdict(int)
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # 分割并处理chunk
        chunk_split = regex.split(pattern, chunk)
        for splited_chunk in chunk_split:
            iter_tokens = regex.finditer(PAT, splited_chunk)
            for token in iter_tokens:
                token_count[token.group(0)] += 1
    
    return dict(token_count)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def parallel_pre_tokenize(
        input_path: str,
        vocab_size: int,
        special_token: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    并行化的BPE预处理函数
    """
    cpu_count = mp.cpu_count()
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")

    special_tokens = special_token
    escaped_tokens = [regex.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    
    # 初始化词汇表
    vocab = {}  
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[len(vocab)] = token.encode("utf-8")
    
    merges = []
    
    # 获取文件块边界
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, cpu_count, "<|endoftext|>".encode("utf-8"))
    
    # 准备并行处理的参数
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((start, end, input_path, pattern))
    
    # 并行处理文件块
    with mp.Pool(processes=cpu_count) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    
    # 合并结果
    token_count = defaultdict(int)
    for chunk_count in chunk_results:
        for token, count in chunk_count.items():
            token_count[token] += count
    
    # 构建字节对计数
    successive = defaultdict(int)
    for token, count in token_count.items():
        token_bytes = token.encode("utf-8")
        if len(token_bytes) > 1:
            for i in range(len(token_bytes) - 1):
                byte_pair = (token_bytes[i], token_bytes[i + 1])
                successive[byte_pair] += count
    
    # BPE合并过程
    while len(vocab) < vocab_size:
        if not successive:
            break
            
        most_common_pair = max(successive, key=successive.get)
        merges.append((vocab[most_common_pair[0]], vocab[most_common_pair[1]]))
        
        index = len(vocab)
        vocab[index] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
        
        del successive[most_common_pair]
        
        # 更新字节对计数
        new_successive = defaultdict(int)
        for (b1, b2), count in successive.items():
            if b1 == most_common_pair[0]:
                new_successive[(most_common_pair[1], b2)] = count
            elif b2 == most_common_pair[1]:
                new_successive[(b1, most_common_pair[0])] = count
            else:
                new_successive[(b1, b2)] = count
        successive = new_successive
    
    return vocab, merges

if __name__ == "__main__":
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"  # Replace with your input file path
    vocab_size = 300  # Example vocabulary size
    special_token = ["<|endoftext|>"]  # Example special token
    vocab, merges = parallel_pre_tokenize(input_path, vocab_size, special_token)
    print("Vocabulary:", vocab)
    print("Merges:", merges)

