#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
__global__ void relu_kernel(const float* __restrict__ in, 
                            float* __restrict__ out, 
                            int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = fmaxf(in[i], 0.0f);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor relu(torch::Tensor x) {
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
    
    relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu, "ReLU activation (CUDA)");
}