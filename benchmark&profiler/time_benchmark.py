import cs336_basics
import argparse
import timeit
import torch
import torch.nn as nn
from typing import Optional
import logging

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(
    vocab_size: int = 10000,
    context_length: int = 2048,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    rope_theta: float = 10000.0,
    device: str = "mps"
) -> BasicsTransformerLM:

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    
    model = model.to(device)
    return model


def generate_random_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # 生成随机目标标签（用于计算损失）
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    return input_ids, targets


def run_forward_pass(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    
    return model(input_ids)


def run_forward_backward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    
    outputs = model(input_ids)
    
    outputs_flat = outputs.view(-1, outputs.size(-1))
    targets_flat = targets.view(-1)
    loss = loss_fn(outputs_flat, targets_flat)
    
    loss.backward()
    
    return outputs, loss


def benchmark_model(
    model: nn.Module,
    batch_size: int = 8,
    seq_length: int = 512,
    vocab_size: int = 32000,
    warmup_steps: int = 5,
    benchmark_steps: int = 10,
    forward_only: bool = False,
    device: str = "cuda"
) -> dict:
    """
    对模型进行基准测试，计算平均值和标准差。
    
    Args:
        model: 要测试的模型
        batch_size: 批次大小
        seq_length: 序列长度
        vocab_size: 词汇表大小
        warmup_steps: 热身步骤数
        benchmark_steps: 基准测试步骤数
        forward_only: 是否只进行前向传播
        device: 设备
        
    Returns:
        包含基准测试结果的字典，包括平均值和标准差
    """
    import numpy as np
    
    model.train()
    
    # 创建优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 生成数据
    input_ids, targets = generate_random_batch(batch_size, seq_length, vocab_size, device)
    
    logger.info(f"开始基准测试: forward_only={forward_only}, warmup_steps={warmup_steps}, benchmark_steps={benchmark_steps}")
    logger.info(f"批次大小: {batch_size}, 序列长度: {seq_length}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {total_params:,}")
    
    # 热身阶段
    if warmup_steps > 0:
        logger.info(f"开始热身 ({warmup_steps} 步)...")
        for i in range(warmup_steps):
            optimizer.zero_grad()
            
            if forward_only:
                _ = run_forward_pass(model, input_ids)
            else:
                _, _ = run_forward_backward_pass(model, input_ids, targets, loss_fn)
            
            # 设备同步
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
    else:
        logger.info("跳过热身阶段")
    
    # 基准测试阶段 - 收集每个步骤的单独计时
    logger.info("开始基准测试...")
    step_times = []
    
    for step in range(benchmark_steps):
        # 记录单步开始时间
        start_time = timeit.default_timer()
        
        optimizer.zero_grad()
        if forward_only:
            _ = run_forward_pass(model, input_ids)
        else:
            _, loss = run_forward_backward_pass(model, input_ids, targets, loss_fn)
        
        # 设备同步
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        
        # 记录单步结束时间
        end_time = timeit.default_timer()
        step_time = end_time - start_time
        step_times.append(step_time)
    
    # 计算统计数据
    step_times = np.array(step_times)
    total_time = np.sum(step_times)
    avg_time_per_step = np.mean(step_times)
    std_time_per_step = np.std(step_times)
    min_time_per_step = np.min(step_times)
    max_time_per_step = np.max(step_times)
    
    # 计算吞吐量
    tokens_per_step = batch_size * seq_length
    tokens_per_second = tokens_per_step / avg_time_per_step
    
    results = {
        "total_time": total_time,
        "avg_time_per_step": avg_time_per_step,
        "std_time_per_step": std_time_per_step,
        "min_time_per_step": min_time_per_step,
        "max_time_per_step": max_time_per_step,
        "step_times": step_times.tolist(),  # 所有单步时间
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_second,
        "steps": benchmark_steps,
        "warmup_steps": warmup_steps,
        "forward_only": forward_only,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "device": device,
        "coefficient_of_variation": std_time_per_step / avg_time_per_step if avg_time_per_step > 0 else 0
    }
    
    return results


def print_results(results: dict):
    """打印基准测试结果，包括平均值和标准差。"""
    print("\n" + "="*70)
    print("基准测试结果")
    print("="*70)
    print(f"模式: {'仅前向传播' if results['forward_only'] else '前向+后向传播'}")
    print(f"设备: {results['device']}")
    print(f"批次大小: {results['batch_size']}")
    print(f"序列长度: {results['seq_length']}")
    print(f"热身步数: {results['warmup_steps']}")
    print(f"测量步数: {results['steps']}")
    print("-"*70)
    print(f"总时间: {results['total_time']:.4f} 秒")
    print(f"平均每步时间: {results['avg_time_per_step']*1000:.2f} ± {results['std_time_per_step']*1000:.2f} 毫秒")
    print(f"最短步时间: {results['min_time_per_step']*1000:.2f} 毫秒")
    print(f"最长步时间: {results['max_time_per_step']*1000:.2f} 毫秒")
    print(f"变异系数 (CV): {results['coefficient_of_variation']:.3f}")
    print(f"每步处理tokens: {results['tokens_per_step']:,}")
    print(f"吞吐量: {results['tokens_per_second']:,.0f} tokens/秒")
    
    # 显示时间变异性分析
    cv = results['coefficient_of_variation']
    if cv < 0.05:
        variability = "很低"
    elif cv < 0.1:
        variability = "低"
    elif cv < 0.2:
        variability = "中等"
    else:
        variability = "高"
    
    print(f"时间变异性: {variability} (标准差/平均值 = {cv:.3f})")
    print("="*70)


def run_warmup_analysis(
    model: nn.Module,
    batch_size: int = 8,
    seq_length: int = 512,
    vocab_size: int = 32000,
    benchmark_steps: int = 10,
    device: str = "cuda"
):
   
    warmup_configs = [0, 1, 2, 5]
    
    print("\n" + "="*80)
    print("热身步骤数影响分析")
    print("="*80)
    
    results_summary = []
    
    for forward_only in [True, False]:
        mode = "仅前向传播" if forward_only else "前向+后向传播"
        print(f"\n【{mode}】")
        print("-" * 40)
        
        for warmup_steps in warmup_configs:
            print(f"\n热身步数: {warmup_steps}")
            results = benchmark_model(
                model=model,
                batch_size=batch_size,
                seq_length=seq_length,
                vocab_size=vocab_size,
                warmup_steps=warmup_steps,
                benchmark_steps=benchmark_steps,
                forward_only=forward_only,
                device=device
            )
            
            results_summary.append({
                'mode': mode,
                'warmup_steps': warmup_steps,
                'avg_time': results['avg_time_per_step'],
                'std_time': results['std_time_per_step'],
                'cv': results['coefficient_of_variation']
            })
            
            print(f"  平均时间: {results['avg_time_per_step']*1000:.2f} ± {results['std_time_per_step']*1000:.2f} ms")
            print(f"  变异系数: {results['coefficient_of_variation']:.3f}")
    
    # 总结分析
    print("\n" + "="*80)
    print("热身步骤数影响总结")
    print("="*80)
    
    for mode in ["仅前向传播", "前向+后向传播"]:
        mode_results = [r for r in results_summary if r['mode'] == mode]
        print(f"\n【{mode}】")
        
        print("热身步数\t平均时间(ms)\t标准差(ms)\t变异系数")
        print("-" * 50)
        for r in mode_results:
            print(f"{r['warmup_steps']}\t\t{r['avg_time']*1000:.2f}\t\t{r['std_time']*1000:.2f}\t\t{r['cv']:.3f}")
        
        # 分析
        no_warmup = mode_results[0]  # 0 warmup steps
        with_warmup = mode_results[-1]  # 5 warmup steps
        
        time_diff = (no_warmup['avg_time'] - with_warmup['avg_time']) * 1000
        cv_diff = no_warmup['cv'] - with_warmup['cv']
        
        print(f"\n无热身 vs 5步热身:")
        print(f"  平均时间差异: {time_diff:.2f} ms ({time_diff/with_warmup['avg_time']/10:.1f}%)")
        print(f"  变异系数差异: {cv_diff:.3f}")
        
        if cv_diff > 0.05:
            print(f"    无热身时变异性明显更高")
        elif time_diff > 1:
            print(f"  ️  无热身时平均时间明显更长")
        else:
            print(f"   热身对该配置影响较小")



def main():
    parser = argparse.ArgumentParser(description="Transformer模型基准测试")
    
    # 模型超参数
    parser.add_argument("--vocab-size", type=int, default=10000, help="词汇表大小")
    parser.add_argument("--context-length", type=int, default=2048, help="上下文长度")
    parser.add_argument("--d-model", type=int, default=768, help="模型维度")
    parser.add_argument("--num-layers", type=int, default=12, help="Transformer层数")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=3072, help="前馈网络维度")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta参数")
    
    # 基准测试参数
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--seq-length", type=int, default=512, help="序列长度")
    parser.add_argument("--warmup-steps", type=int, default=5, help="热身步骤数")
    parser.add_argument("--benchmark-steps", type=int, default=100, help="基准测试步骤数")
    parser.add_argument("--forward-only", action="store_true", help="是否只进行前向传播")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu","mps"], help="设备")
    
    # 新增选项
    parser.add_argument("--warmup-analysis", action="store_true", help="运行热身步骤数影响分析")
    parser.add_argument("--model-size-benchmark", action="store_true", help="运行不同模型大小的基准测试(§1.1.2)")
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，切换到MPS或CPU")
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS不可用，切换到CPU")
        args.device = "cpu"
    
    elif args.warmup_analysis:
        logger.info("创建模型...")
        model = create_model(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=args.device
        )
        
        run_warmup_analysis(
            model=model,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            benchmark_steps=args.benchmark_steps,
            device=args.device
        )
    else:
        # 运行单次基准测试
        logger.info("创建模型...")
        model = create_model(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=args.device
        )
        
        results = benchmark_model(
            model=model,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            forward_only=args.forward_only,
            device=args.device
        )
        
        # 打印结果
        print_results(results)
    
    # 运行热身分析
    run_warmup_analysis(
        model=model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        benchmark_steps=args.benchmark_steps,
        device=args.device
    )
    


if __name__ == "__main__":
    main()

