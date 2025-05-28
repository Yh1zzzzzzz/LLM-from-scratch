#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(const float* __restrict__ in, 
                           float* __restrict__ out, 
                           int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x = in[i];
        const float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + cube)));
    }
}

__global__ void gelu_backward_kernel(const float* __restrict__ grad_output,
                                   const float* __restrict__ input,
                                   float* __restrict__ grad_input,
                                   int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x = input[i];
        const float cube = 0.044715f * x * x * x;
        const float tanh_arg = 0.79788456f * (x + cube);
        const float tanh_val = tanhf(tanh_arg);
        const float sech2 = 1.0f - tanh_val * tanh_val;  // sech^2(x) = 1 - tanh^2(x)
        
        // d/dx[GELU(x)] = 0.5 * (1 + tanh(tanh_arg)) + 0.5 * x * sech^2(tanh_arg) * d/dx[tanh_arg]
        // d/dx[tanh_arg] = 0.79788456 * (1 + 3 * 0.044715 * x^2)
        const float dtanh_arg_dx = 0.79788456f * (1.0f + 3.0f * 0.044715f * x * x);
        const float gelu_grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * dtanh_arg_dx;
        
        grad_input[i] = grad_output[i] * gelu_grad;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    torch::Tensor y = torch::empty_like(x);
    
    const int num_elements = x.numel();
    if (num_elements == 0) return y;  

    const int block_size = 256;  
    const int num_blocks = cdiv(num_elements, block_size);

    const dim3 grid(num_blocks);
    const dim3 block(block_size);
    
    gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return y;
}

torch::Tensor gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(grad_output.is_cuda(), "Grad output tensor must be on CUDA device");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(grad_output.is_contiguous(), "Grad output tensor must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "Grad output and input must have same shape");

    torch::Tensor grad_input = torch::empty_like(input);
    
    const int num_elements = input.numel();
    if (num_elements == 0) return grad_input;

    const int block_size = 256;
    const int num_blocks = cdiv(num_elements, block_size);

    const dim3 grid(num_blocks);
    const dim3 block(block_size);
    
    gelu_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu", &gelu, "GELU activation (CUDA)");
    m.def("gelu_backward", &gelu_backward, "GELU backward pass (CUDA)");
}
/**
 * 
 * HOW TO USE:
import torch
from torch.utils.cpp_extension import load

# 编译CUDA扩展
gelu_extension = load(name="gelu", sources=["cuda_gelu.cu"])

# 定义支持自动求导的GELU函数
class GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = gelu_extension.gelu(input)
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = gelu_extension.gelu_backward(grad_output, input)
        return grad_input

def gelu_cuda(x):
    return GeluFunction.apply(x)

# 使用示例
x = torch.randn(10000, device='cuda', requires_grad=True)
y = gelu_cuda(x)
loss = y.sum()
loss.backward()  # 现在支持反向传播
print(f"Input grad shape: {x.grad.shape}")
 * 
 */