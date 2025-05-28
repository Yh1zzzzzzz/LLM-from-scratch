import torch
import triton
import triton.language as tl

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    # Read
    x = tl.load(x_ptr + offsets, mask=mask)
    #  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()
    
    y = torch.empty_like(x)

    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y

@triton.jit
def triton_gelu_backward_kernel(grad_output_ptr, input_ptr, grad_input_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    """GELU反向传播的Triton核函数"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < num_elements
    
    # 读取梯度输出和输入
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算GELU的导数
    cube = 0.044715 * x * x * x
    tanh_arg = 0.79788456 * (x + cube)
    exp_2a = tl.exp(2 * tanh_arg)
    tanh_val = (exp_2a - 1) / (exp_2a + 1)
    sech2 = 1.0 - tanh_val * tanh_val
    
    # d/dx[tanh_arg] = 0.79788456 * (1 + 3 * 0.044715 * x^2)
    dtanh_arg_dx = 0.79788456 * (1.0 + 3.0 * 0.044715 * x * x)
    
    # GELU的导数
    gelu_grad = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dtanh_arg_dx
    
    # 计算输入梯度
    grad_input = grad_output * gelu_grad
    
    # 存储结果
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def triton_gelu_backward(grad_output: torch.Tensor, input: torch.Tensor):
    """GELU反向传播的Triton实现"""
    assert grad_output.is_cuda and input.is_cuda
    assert grad_output.is_contiguous() and input.is_contiguous()
    assert grad_output.shape == input.shape
    
    grad_input = torch.empty_like(input)
    
    num_elements = input.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)
    
    triton_gelu_backward_kernel[(num_blocks,)](
        grad_output, input, grad_input, num_elements, BLOCK_SIZE=block_size
    )
    
    return grad_input


class TritonGELUFunction(torch.autograd.Function):
    """支持自动求导的Triton GELU函数"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return triton_gelu(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return triton_gelu_backward(grad_output, input)


def gelu_triton(x):
    """支持自动求导的Triton GELU接口"""
    return TritonGELUFunction.apply(x)


if __name__ == "__main__":
    import time
    
    print("Triton GELU激活函数测试")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print(" CUDA不可用，无法运行Triton测试")
        exit(1)
    
    # 测试数据
    x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
    
    # 测试前向传播和反向传播
    print("测试前向传播和反向传播...")
    y = gelu_triton(x)
    loss = y.sum()
    loss.backward()
    
    print(f" 输入形状: {x.shape}")
    print(f" 输出形状: {y.shape}")
    print(f"梯度形状: {x.grad.shape}")
    
    # 性能测试
    print("\n性能测试...")
    x_test = torch.randn(2048, 2048, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = triton_gelu(x_test)
    
    torch.cuda.synchronize()
    
    # Triton版本
    start_time = time.time()
    for _ in range(100):
        y_triton = triton_gelu(x_test)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / 100
    
    # PyTorch版本
    gelu_torch = torch.nn.GELU()
    start_time = time.time()
    for _ in range(100):
        y_torch = gelu_torch(x_test)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / 100
    
    print(f"Triton GELU: {triton_time*1000:.3f} ms")
    print(f"PyTorch GELU: {torch_time*1000:.3f} ms")
    print(f"加速比: {torch_time/triton_time:.2f}x")
    
    # 精度对比
    diff = torch.abs(y_triton - y_torch).max().item()
    print(f"最大差异: {diff:.8f}")
    
    print("\n测试完成!")