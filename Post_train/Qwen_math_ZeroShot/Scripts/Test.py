# (a) 编写一个脚本，用于评估 Qwen 2.5 Math 1.5B 模型在 MATH 数据集上的零样本（zero-shot）性能。该脚本应：
# (1) 加载 MATH 验证集样本。
# (2) 使用 r1_zero 提示模板将这些样本格式化为字符串提示（prompts），以输入给语言模型。
# (3) 为每个样本生成输出。
# 该脚本还应：
# (4) 计算评估指标。
# (5) 将样本、模型生成的内容以及相应的评估分数序列化到磁盘，以便在后续问题中进行分析。
    # 在您的实现中，包含一个参数与下面类似的方法 evaluate_vllm 可能会很有帮助，因为您之后可以复用它：
from typing import Callable, List
from vllm import LLM, SamplingParams
import json
from DrGrpo import r1_zero_reward_fn
import random
from datasets import Dataset
import pyarrow.parquet as pq
from pathlib import Path


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers: List[str]
) -> None:
    """
    评估语言模型在一系列提示（prompts）上的表现，
    计算评估指标，并将结果序列化到磁盘。
    """
    sample_param = eval_sampling_params
    prompt_outputs = vllm_model.generate(prompts, sampling_params=sample_param)
    results = []
    for i, output in enumerate(prompt_outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, answers[i])
        results.append({
            "prompt": prompt,
            "output": generated_text,
            "reward": reward,
        })
    with open("evaluation_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n" + '\n')
            f.write('\n')


    
def build_prompt(template_path: str, question: str) -> str:
    with open(template_path, 'r') as f:
        template = f.read()
    return template.replace("{question}", question)


#参数加载
BASE_DIR = Path(__file__).resolve().parent
prompt = BASE_DIR /   "Prompts" / "r1_zero.prompt"
math_dataset_path = BASE_DIR / "data" / "tulu-3-sft-personas-math" / "data" / "train-00000-of-00002.parquet"
LLM_model = BASE_DIR / "data" / "models" / "Qwen2.5-Math-1.5B"

#vllm参数设置
sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=1024)
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True


if __name__ == "__main__":
    print(BASE_DIR) 

#构建 dataset 、prompts 和 answers
    table = pq.read_table(math_dataset_path)
    ds = Dataset(arrow_table=table)
    random.seed(114514)
    sample_size = 20
    selected_indices = random.sample(range(len(ds)), k=sample_size)
    sampled_examples = [ds[i] for i in selected_indices]
    prompts = [item['prompt'] for item in sampled_examples]
    answers = [item['messages'] for item in sampled_examples]

    formatted_prompts = [build_prompt(prompt, q) for q in prompts]
    llm_model = LLM(model=str(LLM_model))
    evaluate_vllm(
        vllm_model=llm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=formatted_prompts,
        eval_sampling_params=sampling_params,
        answers=answers
    )


