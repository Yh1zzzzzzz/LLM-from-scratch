#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton ReLU激活函数实现
包含前向传播和反向传播
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_relu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    """ReLU前向传播的Triton核函数"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    
    # 读取输入
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # ReLU: max(0, x)
    y = tl.maximum(x, 0.0)
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def triton_relu_backward_kernel(grad_output_ptr, input_ptr, grad_input_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    """ReLU反向传播的Triton核函数"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    
    # 读取梯度输出和输入
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # ReLU的导数: 1 if x > 0 else 0
    grad_input = tl.where(x > 0.0, grad_output, 0.0)
    
    # 存储结果
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def triton_relu(x: torch.Tensor):
    """ReLU前向传播的Triton实现"""
    assert x.is_cuda
    assert x.is_contiguous()
    
    y = torch.empty_like(x)

    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_relu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y


def triton_relu_backward(grad_output: torch.Tensor, input: torch.Tensor):
    """ReLU反向传播的Triton实现"""
    assert grad_output.is_cuda and input.is_cuda
    assert grad_output.is_contiguous() and input.is_contiguous()
    assert grad_output.shape == input.shape
    
    grad_input = torch.empty_like(input)
    
    num_elements = input.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_relu_backward_kernel[(num_blocks,)](
        grad_output, input, grad_input, num_elements, BLOCK_SIZE=block_size
    )
    
    return grad_input


class TritonReLUFunction(torch.autograd.Function):
    """支持自动求导的Triton ReLU函数"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return triton_relu(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return triton_relu_backward(grad_output, input)


def relu_triton(x):
    """支持自动求导的Triton ReLU接口"""
    return TritonReLUFunction.apply(x)


if __name__ == "__main__":
    import time
    
    print("Triton ReLU激活函数测试")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print(" CUDA不可用，无法运行Triton测试")
        exit(1)
    
    # 测试数据
    x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
    
    # 测试前向传播和反向传播
    print("测试前向传播和反向传播...")
    y = relu_triton(x)
    loss = y.sum()
    loss.backward()
    
    print(f" 输入形状: {x.shape}")
    print(f" 输出形状: {y.shape}")
    print(f" 梯度形状: {x.grad.shape}")
    
    # 精度验证
    print("\n精度验证...")
    x_test = torch.randn(100, 100, device='cuda')
    y_triton = triton_relu(x_test)
    y_torch = torch.relu(x_test)
    
    diff = torch.abs(y_triton - y_torch).max().item()
    print(f"与PyTorch ReLU的最大差异: {diff:.8f}")
    
    if diff < 1e-6:
        print(" 精度验证通过!")
    else:
        print(" 精度验证失败!")
    
    # 性能测试
    print("\n性能测试...")
    x_perf = torch.randn(2048, 2048, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = triton_relu(x_perf)
        _ = torch.relu(x_perf)
    
    torch.cuda.synchronize()
    
    # Triton版本
    start_time = time.time()
    for _ in range(100):
        y_triton = triton_relu(x_perf)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 100
    
    # PyTorch版本
    start_time = time.time()
    for _ in range(100):
        y_torch = torch.relu(x_perf)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / 100
    
    print(f"Triton ReLU: {triton_time*1000:.3f} ms")
    print(f"PyTorch ReLU: {torch_time*1000:.3f} ms")
    print(f"加速比: {torch_time/triton_time:.2f}x")
    
    print("\n测试完成!")