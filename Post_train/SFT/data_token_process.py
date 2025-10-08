from transformers import AutoTokenizer
import torch

# 训练数据整理，方便训练:
# 形状如下：
# prompt: p
# output: o

# 用这一行来对下一行做next token prediction: 
# (input_id)p1 p2 p3 p4 o1 o2 o3 o4    [pad]
# (label)   p2 p3 p4 o1 o2 o3 o4 [pad] [pad] 通常我们会舍弃最后一个没用的，因为无需对o4(最后一个生成的token)做预测                     


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    # Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    # other tokens (prompt or padding).
    # Args:
    # prompt_strs: list[str] List of prompt strings.
    # output_strs: list[str] List of output strings.
    # tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    # Returns:
    # dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
    # the tokenized prompt and output strings. Then the returned dictionary should have the
    # following keys:
    # input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    # the tokenized prompt and output strings, with the final token sliced off.
    # labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    # shifted input ids, i.e., the input ids without the first token.
    # response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
    # 1): a mask on the response tokens in the labels
    def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_tokens = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    concat_tokens = [p + o for p, o in zip(prompt_tokens, output_tokens)]
    max_len = max(len(t) for t in concat_tokens)
    dic = {}
    input_ids = []
    labels = []
    response_mask = []
    for pt, ot in zip(prompt_tokens, output_tokens):
        concat = pt + ot
        pad_len = max_len - len(concat)
        input_id = concat + [tokenizer.pad_token_id] * pad_len
        label = input_id[1:] + [tokenizer.pad_token_id]
        response_m = [False] * (len(pt) - 1) + [True] * len(ot) + [False] * (pad_len+1)
        input_ids.append(torch.tensor(input_id[:-1], dtype=torch.long))
        labels.append(torch.tensor(label[:-1], dtype=torch.long))
        response_mask.append(torch.tensor(response_m[:-1], dtype=torch.long))
    dic['input_ids'] = torch.stack(input_ids)
    dic['labels'] = torch.stack(labels)
    dic['response_mask'] = torch.stack(response_mask)
    return dic
        