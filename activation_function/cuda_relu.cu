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

__global__ void relu_backward_kernel(const float* __restrict__ grad_output,
                                   const float* __restrict__ input,
                                   float* __restrict__ grad_input,
                                   int num_elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
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

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input) {
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
    
    relu_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        num_elements
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return grad_input;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu, "ReLU activation (CUDA)");
    m.def("relu_backward", &relu_backward, "ReLU backward pass (CUDA)");
}