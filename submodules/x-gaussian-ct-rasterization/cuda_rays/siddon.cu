#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

template <typename T, bool backward>
__global__ void integrate(const T* volume, const T* gradient, T* output, T* grad_volume,
    const T* origin, const T* directions, const T* low, const T* spacing,
    int nx, int ny, int nz, int rays) {
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= rays) return;
    const int dims[3] = {nx, ny, nz};
    T near = T(0), far = T(1e30), direction[3], inverse[3];
    for (int axis=0; axis<3; ++axis) {
        direction[axis] = directions[3*ray+axis];
        inverse[axis] = fabs(direction[axis]) > T(1e-8) ? T(1)/direction[axis] : T(1e8);
        T a = (low[axis]-origin[axis])*inverse[axis];
        T b = (low[axis]+spacing[axis]*dims[axis]-origin[axis])*inverse[axis];
        near = max(near, min(a,b));
        far = min(far, max(a,b));
    }
    if (far <= near || (backward && gradient[ray] == T(0))) {
        if (!backward) output[ray] = T(0);
        return;
    }
    int index[3], sign[3];
    T next[3];
    for (int axis=0; axis<3; ++axis) {
        T entry = origin[axis]+near*direction[axis];
        index[axis] = min(dims[axis]-1, max(0,int(floor((entry-low[axis])/spacing[axis]))));
        sign[axis] = direction[axis]>T(1e-8) ? 1 : (direction[axis]<T(-1e-8) ? -1 : 0);
        next[axis] = sign[axis] ?
            (low[axis]+T(index[axis]+(sign[axis]>0))*spacing[axis]-origin[axis])*inverse[axis] : T(1e30);
    }
    T current=near, sum=T(0);
    for (int iteration=0; iteration<nx+ny+nz+8; ++iteration) {
        T end = min(far,min(next[0],min(next[1],next[2])));
        T length = max(T(0),end-current);
        long long flat = (static_cast<long long>(index[0])*ny+index[1])*nz+index[2];
        if (length > T(0)) {
            if (backward) atomicAdd(grad_volume+flat, gradient[ray]*length);
            else sum += volume[flat]*length;
        }
        if (end >= far) break;
        bool outside=false;
        for (int axis=0; axis<3; ++axis) {
            if (next[axis] <= end) {
                index[axis] += sign[axis];
                if (index[axis]<0 || index[axis]>=dims[axis]) outside=true;
                next[axis] = sign[axis] ?
                    (low[axis]+T(index[axis]+(sign[axis]>0))*spacing[axis]-origin[axis])*inverse[axis] : T(1e30);
            }
        }
        if (outside) break;
        current=max(current,end);
    }
    if (!backward) output[ray]=sum;
}

void check(torch::Tensor volume, torch::Tensor origin, torch::Tensor directions,
           torch::Tensor low, torch::Tensor spacing) {
    TORCH_CHECK(volume.is_cuda() && volume.is_contiguous() && volume.dim()==3,
                "Expected a contiguous CUDA volume");
    TORCH_CHECK(volume.scalar_type()==torch::kFloat32 || volume.scalar_type()==torch::kFloat64,
                "Expected float32 or float64");
    TORCH_CHECK(directions.dim()==2 && directions.size(1)==3,"Expected N x 3 directions");
    TORCH_CHECK(origin.numel()==3 && low.numel()==3 && spacing.numel()==3,"Expected 3D geometry");
    for (const auto& tensor : {origin,directions,low,spacing}) {
        TORCH_CHECK(tensor.device()==volume.device() && tensor.scalar_type()==volume.scalar_type()
                    && tensor.is_contiguous(), "Geometry must match volume device/dtype/layout");
    }
}

torch::Tensor siddon_forward_cuda(torch::Tensor volume, torch::Tensor origin,
    torch::Tensor directions, torch::Tensor low, torch::Tensor spacing) {
    check(volume,origin,directions,low,spacing);
    c10::cuda::CUDAGuard guard(volume.device());
    auto out=torch::empty({directions.size(0)},volume.options());
    if (!out.numel()) return out;
    int blocks=(out.numel()+255)/256;
    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(),"siddon_forward",[&]{
        integrate<scalar_t,false><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
            volume.data_ptr<scalar_t>(),nullptr,out.data_ptr<scalar_t>(),nullptr,
            origin.data_ptr<scalar_t>(),directions.data_ptr<scalar_t>(),low.data_ptr<scalar_t>(),
            spacing.data_ptr<scalar_t>(),volume.size(0),volume.size(1),volume.size(2),out.numel());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor siddon_backward_cuda(torch::Tensor grad, torch::Tensor volume,
    torch::Tensor origin, torch::Tensor directions, torch::Tensor low, torch::Tensor spacing) {
    check(volume,origin,directions,low,spacing);
    TORCH_CHECK(grad.device()==volume.device() && grad.scalar_type()==volume.scalar_type()
                && grad.is_contiguous() && grad.numel()==directions.size(0),"Invalid output gradient");
    c10::cuda::CUDAGuard guard(volume.device());
    auto out=torch::zeros_like(volume);
    if (!grad.numel()) return out;
    int blocks=(grad.numel()+255)/256;
    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(),"siddon_backward",[&]{
        integrate<scalar_t,true><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
            nullptr,grad.data_ptr<scalar_t>(),nullptr,out.data_ptr<scalar_t>(),
            origin.data_ptr<scalar_t>(),directions.data_ptr<scalar_t>(),low.data_ptr<scalar_t>(),
            spacing.data_ptr<scalar_t>(),volume.size(0),volume.size(1),volume.size(2),grad.numel());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
