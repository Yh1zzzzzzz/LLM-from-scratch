#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton SwiGLU激活函数实现
包含前向传播和反向传播
SwiGLU(x, gate) = Swish(x) * gate
其中 Swish(x) = x * sigmoid(x)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_swiglu_kernel(x_ptr, gate_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    """SwiGLU前向传播的Triton核函数"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    
    # 读取输入
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)
    
    # Swish(x) = x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    swish_x = x * sigmoid_x
    
    # SwiGLU = Swish(x) * gate
    y = swish_x * gate
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def triton_swiglu_backward_kernel(grad_output_ptr, x_ptr, gate_ptr, 
                                 grad_x_ptr, grad_gate_ptr, num_elements, 
                                 BLOCK_SIZE: tl.constexpr):
    """SwiGLU反向传播的Triton核函数"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    
    # 读取梯度输出和输入
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)
    
    # 计算Swish及其导数
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    swish_x = x * sigmoid_x
    # Swish的导数: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    swish_grad = sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x)
    
    # SwiGLU对x的梯度
    grad_x = grad_output * gate * swish_grad
    
    # SwiGLU对gate的梯度
    grad_gate = grad_output * swish_x
    
    # 存储结果
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)


def triton_swiglu(x: torch.Tensor, gate: torch.Tensor):
    """SwiGLU前向传播的Triton实现"""
    assert x.is_cuda and gate.is_cuda
    assert x.is_contiguous() and gate.is_contiguous()
    assert x.shape == gate.shape
    
    y = torch.empty_like(x)

    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_swiglu_kernel[(num_blocks,)](x, gate, y, num_elements, BLOCK_SIZE=block_size)
    return y


def triton_swiglu_backward(grad_output: torch.Tensor, x: torch.Tensor, gate: torch.Tensor):
    """SwiGLU反向传播的Triton实现"""
    assert grad_output.is_cuda and x.is_cuda and gate.is_cuda
    assert grad_output.is_contiguous() and x.is_contiguous() and gate.is_contiguous()
    assert grad_output.shape == x.shape == gate.shape
    
    grad_x = torch.empty_like(x)
    grad_gate = torch.empty_like(gate)
    
    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_swiglu_backward_kernel[(num_blocks,)](
        grad_output, x, gate, grad_x, grad_gate, num_elements, BLOCK_SIZE=block_size
    )
    
    return grad_x, grad_gate


class TritonSwiGLUFunction(torch.autograd.Function):
    """支持自动求导的Triton SwiGLU函数"""
    
    @staticmethod
    def forward(ctx, x, gate):
        ctx.save_for_backward(x, gate)
        return triton_swiglu(x, gate)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        return triton_swiglu_backward(grad_output, x, gate)


def swiglu_triton(x, gate):
    """支持自动求导的Triton SwiGLU接口"""
    return TritonSwiGLUFunction.apply(x, gate)


def swish_pytorch(x):
    """PyTorch版本的Swish激活函数，用于对比"""
    return x * torch.sigmoid(x)


def swiglu_pytorch(x, gate):
    """PyTorch版本的SwiGLU激活函数，用于对比"""
    return swish_pytorch(x) * gate


if __name__ == "__main__":
    import time
    
    print("Triton SwiGLU激活函数测试")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print(" CUDA不可用，无法运行Triton测试")
        exit(1)
    
    # 测试数据
    x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
    gate = torch.randn(1024, 1024, device='cuda', requires_grad=True)
    
    # 测试前向传播和反向传播
    print("测试前向传播和反向传播...")
    y = swiglu_triton(x, gate)
    loss = y.sum()
    loss.backward()
    
    print(f" 输入x形状: {x.shape}")
    print(f" 输入gate形状: {gate.shape}")
    print(f" 输出形状: {y.shape}")
    print(f" x梯度形状: {x.grad.shape}")
    print(f" gate梯度形状: {gate.grad.shape}")
    
    # 精度验证
    print("\n精度验证...")
    x_test = torch.randn(100, 100, device='cuda')
    gate_test = torch.randn(100, 100, device='cuda')
    
    y_triton = triton_swiglu(x_test, gate_test)
    y_pytorch = swiglu_pytorch(x_test, gate_test)
    
    diff = torch.abs(y_triton - y_pytorch).max().item()
    print(f"与PyTorch SwiGLU的最大差异: {diff:.8f}")
    
    if diff < 1e-4:
        print(" 精度验证通过!")
    else:
        print(" 精度验证失败!")
    
    # 性能测试
    print("\n性能测试...")
    x_perf = torch.randn(2048, 2048, device='cuda')
    gate_perf = torch.randn(2048, 2048, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = triton_swiglu(x_perf, gate_perf)
        _ = swiglu_pytorch(x_perf, gate_perf)
    
    torch.cuda.synchronize()
    
    # Triton版本
    start_time = time.time()
    for _ in range(100):
        y_triton = triton_swiglu(x_perf, gate_perf)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 100
    
    # PyTorch版本
    start_time = time.time()
    for _ in range(100):
        y_pytorch = swiglu_pytorch(x_perf, gate_perf)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / 100
    
    print(f"Triton SwiGLU: {triton_time*1000:.3f} ms")
    print(f"PyTorch SwiGLU: {pytorch_time*1000:.3f} ms")
    print(f"加速比: {pytorch_time/triton_time:.2f}x")
    
    print("\n测试完成!")