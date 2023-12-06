#include "sgmv_cutlass.cuh"

template bool sgmv<half>(nv_half *y, nv_half *x, nv_half **w, int32_t *s,
                            void *tmp_d, int num_problems, int d_in, int d_out,
                            int layer_idx, cudaStream_t stream);

template bool sgmv<float>(float *y, float *x, float **w, int32_t *s,
                            void *tmp_d, int num_problems, int d_in, int d_out,
                            int layer_idx, cudaStream_t stream);

#ifdef ENABLE_BF16
template bool sgmv<nv_bfloat16>(nv_bfloat16 *y, nv_bfloat16 *x, nv_bfloat16 **w,
                                int32_t *s, void *tmp_d, int num_problems,
                                int d_in, int d_out, int layer_idx, cudaStream_t stream);
#endif