import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize(text: str) -> list[str]:
    return regex.findall(PAT, text)
def _pre_token_iterate(text: str):
    return regex.finditer(PAT, text)
string = "This is a test string with numbers 123 and special characters !@#."
if __name__ == "__main__":
    tokens = pre_tokenize(string)
    print(tokens)
