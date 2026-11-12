#include <torch/extension.h>

torch::Tensor siddon_forward_cuda(torch::Tensor volume, torch::Tensor origin,
    torch::Tensor directions, torch::Tensor low, torch::Tensor spacing);
torch::Tensor siddon_backward_cuda(torch::Tensor grad, torch::Tensor volume,
    torch::Tensor origin, torch::Tensor directions, torch::Tensor low, torch::Tensor spacing);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &siddon_forward_cuda);
    m.def("backward", &siddon_backward_cuda);
}
