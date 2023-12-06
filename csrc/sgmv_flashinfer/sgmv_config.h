#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T, uint32_t d_out>
bool sgmv_shrink(T* y, const T* x, T** w, const int32_t* s, void* tmp,
                 uint32_t num_problems, uint32_t d_in, uint32_t layer_idx, cudaStream_t stream);

// clang-format off

#define FOR_SGMV_NARROW(f, T) \
    f(T, 16) \
    f(T, 32) \
    f(T, 64) \
    f(T, 96) \
    f(T, 128) \
    f(T, 160) \
    f(T, 192)

// clang-format on
