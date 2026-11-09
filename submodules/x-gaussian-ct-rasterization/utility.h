#ifndef AUXILIARY_H_INCLUDED
#define AUXILIARY_H_INCLUDED

#include <functional>
#include <torch/extension.h>

inline std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

#endif
