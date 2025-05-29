import os
from typing import BinaryIO
import io
import regex
import heapq
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize(text: str) -> list[str]:
    return regex.findall(PAT, text)
def pre_token_iterate(text: str):
    return regex.finditer(PAT, text)

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



def serial_pre_tokenize(
        input_path: str,
        vocab_size: int,
        special_token: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Pre-tokenize the input file and return a vocabulary and merges.
    """
    token_count = {} #字符级别的计数表
    cpu_count = mp.cpu_count()
    path = input_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    special_tokens = special_token # 可以包含多个特殊标记
    escaped_tokens = [regex.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)  # 正确的模式，如 "\<\|endoftext\|\>"
    vocab = {}  
    for i in range(256):
        vocab[i] = bytes([i])  # Initialize vocab with single byte characters
    for i in range(len(special_tokens)):
        len_vocab = len(vocab)
        vocab[len_vocab] = special_tokens[i].encode("utf-8")  # Add special tokens to vocab
    merges = [] # list[tuple[byte, byte]]  # List of merges, if needed
    successive = {}


    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, cpu_count, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_split = regex.split(pattern, chunk)
            for splited_chunk in chunk_split:
                iter_tokens = pre_token_iterate(splited_chunk)
                for token in iter_tokens:
                    if token.group(0) not in token_count:
                        token_count[token.group(0)] = 0
                    token_count[token.group(0)] += 1
    print(sorted(token_count.items(), key=lambda x: x[1], reverse=True))
   
    for token, count in token_count.items():
        token_bytes = token.encode("utf-8")  # 转换为字节
        if len(token_bytes) > 1:
            for i in range(len(token_bytes) - 1):
                byte_pair = (token_bytes[i], token_bytes[i + 1])  # 直接使用字节值
                if byte_pair not in successive:
                    successive[byte_pair] = 0
                successive[byte_pair] += count
    while len(vocab) < vocab_size:
        # Find the most common byte pair
        if not successive:
            break
        most_common_pair = max(successive, key=successive.get)
        merges.append((vocab[most_common_pair[0]], vocab[most_common_pair[1]]))        
        index =  len(vocab) 
        vocab[index] = vocab[most_common_pair[0]] +  vocab[most_common_pair[1]] #most_common_pair
        
        # Remove the most common pair from successive
        del successive[most_common_pair]
        
        # Update successive counts
        new_successive = {}
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
    input_path = "test.txt"  # Replace with your input file path
    vocab_size = 300  # Example vocabulary size
    special_token = ["<|endoftext|>"]  # Example special token
    vocab, merges = serial_pre_tokenize(input_path, vocab_size, special_token)
    print("Vocabulary:", vocab)
    print("Merges:", merges)

