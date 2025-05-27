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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu", &gelu, "GELU activation (CUDA)");
}
/**
 * 
 * HOW TO USE:
import torch
from torch.utils.cpp_extension import load

gelu_extension = load(name="gelu", sources=["gelu.cu"])

x = torch.randn(10000, device='cuda')
y = gelu_extension.gelu(x)
 * 
 */