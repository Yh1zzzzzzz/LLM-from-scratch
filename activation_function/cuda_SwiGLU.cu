#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void swiglu_kernel(const float* __restrict__ x, 
                             const float* __restrict__ gate,
                             float* __restrict__ out, 
                             int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x_val = x[i];
        const float gate_val = gate[i];
        // Swish(x) = x * sigmoid(x)
        const float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        const float swish_x = x_val * sigmoid_x;
        // SwiGLU = Swish(x) * gate
        out[i] = swish_x * gate_val;
    }
}

__global__ void swiglu_backward_kernel(const float* __restrict__ grad_output,
                                      const float* __restrict__ x,
                                      const float* __restrict__ gate,
                                      float* __restrict__ grad_x,
                                      float* __restrict__ grad_gate,
                                      int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const float x_val = x[i];
        const float gate_val = gate[i];
        const float grad_out = grad_output[i];
        
        // 计算Swish及其导数
        const float sigmoid_x = 1.0f / (1.0f + expf(-x_val));
        const float swish_x = x_val * sigmoid_x;
        // Swish的导数: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        const float swish_grad = sigmoid_x + x_val * sigmoid_x * (1.0f - sigmoid_x);
        
        // SwiGLU对x的梯度
        grad_x[i] = grad_out * gate_val * swish_grad;
        
        // SwiGLU对gate的梯度
        grad_gate[i] = grad_out * swish_x;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor swiglu(torch::Tensor x, torch::Tensor gate) {
    TORCH_CHECK(x.is_cuda() && gate.is_cuda(), "Input tensors must be on CUDA device");
    TORCH_CHECK(x.is_contiguous() && gate.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && gate.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(x.sizes() == gate.sizes(), "x and gate must have same shape");

    torch::Tensor y = torch::empty_like(x);
    
    const int num_elements = x.numel();
    if (num_elements == 0) return y;

    const int block_size = 256;
    const int num_blocks = cdiv(num_elements, block_size);

    const dim3 grid(num_blocks);
    const dim3 block(block_size);
    
    swiglu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        gate.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return y;
}

std::vector<torch::Tensor> swiglu_backward(torch::Tensor grad_output, 
                                          torch::Tensor x, 
                                          torch::Tensor gate) {
    TORCH_CHECK(grad_output.is_cuda() && x.is_cuda() && gate.is_cuda(), 
                "All tensors must be on CUDA device");
    TORCH_CHECK(grad_output.is_contiguous() && x.is_contiguous() && gate.is_contiguous(), 
                "All tensors must be contiguous");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32 && gate.dtype() == torch::kFloat32, 
                "Only float32 supported");
    TORCH_CHECK(grad_output.sizes() == x.sizes() && x.sizes() == gate.sizes(), 
                "All tensors must have same shape");

    torch::Tensor grad_x = torch::empty_like(x);
    torch::Tensor grad_gate = torch::empty_like(gate);
    
    const int num_elements = x.numel();
    if (num_elements == 0) return {grad_x, grad_gate};

    const int block_size = 256;
    const int num_blocks = cdiv(num_elements, block_size);

    const dim3 grid(num_blocks);
    const dim3 block(block_size);
    
    swiglu_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        x.data_ptr<float>(),
        gate.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_gate.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return {grad_x, grad_gate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu", &swiglu, "SwiGLU activation (CUDA)");
    m.def("swiglu_backward", &swiglu_backward, "SwiGLU backward pass (CUDA)");
}

/**
 * 
 * HOW TO USE:
import torch
from torch.utils.cpp_extension import load

# 编译CUDA扩展
swiglu_extension = load(name="swiglu", sources=["cuda_SwiGLU.cu"])

# 定义支持自动求导的SwiGLU函数
class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        output = swiglu_extension.swiglu(x, gate)
        ctx.save_for_backward(x, gate)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        grad_x, grad_gate = swiglu_extension.swiglu_backward(grad_output, x, gate)
        return grad_x, grad_gate

def swiglu_cuda(x, gate):
    return SwiGLUFunction.apply(x, gate)

# 使用示例
x = torch.randn(1000, 512, device='cuda', requires_grad=True)
gate = torch.randn(1000, 512, device='cuda', requires_grad=True)
y = swiglu_cuda(x, gate)
loss = y.sum()
loss.backward()  # 现在支持反向传播
print(f"x grad shape: {x.grad.shape}")
print(f"gate grad shape: {gate.grad.shape}")
 * 
 */ 