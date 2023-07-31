#include <torch/extension.h>

#include "gen/punica_ops.cc.inc"

namespace {

//====== utils ======

void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                 const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

//====== rotary_mha_decode ======

template <typename F>
void rotary_mha_decode_kvconst(F kernel, torch::Tensor q_proj,
                               torch::Tensor k_proj, torch::Tensor v_proj,
                               torch::Tensor o, torch::Tensor past_len,
                               torch::Tensor kvbuf, torch::Tensor kvidx,
                               int64_t layer_idx) {
  int64_t B = q_proj.size(0);
  int64_t nnz = kvbuf.size(0);
  kernel(k_proj.data_ptr(), o.data_ptr(), q_proj.data_ptr(), v_proj.data_ptr(),
         kvbuf.data_ptr(), kvidx.data_ptr(), past_len.data_ptr(), B, layer_idx,
         nnz);
}

#define DEFINE_rotary_mha_decode_kvconst(name)                                \
  void name(torch::Tensor q_proj, torch::Tensor k_proj, torch::Tensor v_proj, \
            torch::Tensor o, torch::Tensor past_len, torch::Tensor kvbuf,     \
            torch::Tensor kvidx, int64_t layer_idx) {                         \
    rotary_mha_decode_kvconst(launch_##name##_kernel, q_proj, k_proj, v_proj, \
                              o, past_len, kvbuf, kvidx, layer_idx);          \
  }

template <typename F>
void rotary_mha_decode(F kernel, torch::Tensor q_proj, torch::Tensor k_proj,
                       torch::Tensor v_proj, torch::Tensor o,
                       torch::Tensor past_len, torch::Tensor kvbuf,
                       torch::Tensor kvidx, int64_t layer_idx) {
  int64_t B = q_proj.size(0);
  int64_t H = q_proj.size(1);
  int64_t nnz = kvbuf.size(0);
  int64_t L = kvbuf.size(1);
  int64_t MAXLEN = kvbuf.size(3);
  kernel(k_proj.data_ptr(), o.data_ptr(), q_proj.data_ptr(), v_proj.data_ptr(),
         kvbuf.data_ptr(), kvidx.data_ptr(), past_len.data_ptr(), B, H, L,
         MAXLEN, layer_idx, nnz);
}

#define DEFINE_rotary_mha_decode(name)                                        \
  void name(torch::Tensor q_proj, torch::Tensor k_proj, torch::Tensor v_proj, \
            torch::Tensor o, torch::Tensor past_len, torch::Tensor kvbuf,     \
            torch::Tensor kvidx, int64_t layer_idx) {                         \
    rotary_mha_decode(launch_##name##_kernel, q_proj, k_proj, v_proj, o,      \
                      past_len, kvbuf, kvidx, layer_idx);                     \
  }

void dispatch_rotary_mha_decode(torch::Tensor q_proj, torch::Tensor k_proj,
                                torch::Tensor v_proj, torch::Tensor o,
                                torch::Tensor past_len, torch::Tensor kvbuf,
                                torch::Tensor kvidx, int64_t layer_idx) {
  CHECK_INPUT(q_proj);
  CHECK_INPUT(k_proj);
  CHECK_INPUT(v_proj);
  CHECK_INPUT(o);
  CHECK_INPUT(past_len);
  CHECK_INPUT(kvbuf);
  CHECK_INPUT(kvidx);

  CHECK_DIM(3, q_proj);
  CHECK_DIM(3, k_proj);
  CHECK_DIM(3, v_proj);
  CHECK_DIM(3, o);
  CHECK_DIM(1, past_len);
  CHECK_DIM(6, kvbuf);
  CHECK_DIM(1, kvidx);

  int64_t B = q_proj.size(0);
  int64_t H = q_proj.size(1);
  int64_t D = q_proj.size(2);
  CHECK_SHAPE(q_proj, k_proj);
  CHECK_SHAPE(q_proj, v_proj);
  CHECK_SHAPE(q_proj, o);
  TORCH_CHECK(past_len.size(0) == B, "past_len.size(0) != B. ",
              past_len.size(0), " vs ", B);

  int64_t L = kvbuf.size(1);
  TORCH_CHECK(kvbuf.size(2) == 2, "kvbuf.size(2) != 2. Actual:", kvbuf.size(2));
  int64_t maxlen = kvbuf.size(3);
  TORCH_CHECK(kvbuf.size(4) == H, "kvbuf.size(4) != H. ", kvbuf.size(4), " vs ",
              H);
  TORCH_CHECK(kvbuf.size(5) == D, "kvbuf.size(5) != D. ", kvbuf.size(5), " vs ",
              D);
  TORCH_CHECK(kvidx.size(0) == B, "kvidx.size(0) != B. ", kvidx.size(0), " vs ",
              B);

#define DISPATCH(num_heads, head_dim, num_layers, MAXLEN, dtype)                                                \
  if (H == num_heads && D == head_dim && L == num_layers &&                                                     \
      maxlen == MAXLEN) {                                                                                       \
    return rotary_mha_decode_kvconst(                                                                           \
        launch_rotary_mha_decode_kvconst_##num_heads##_##head_dim##_##num_layers##_##MAXLEN##_##dtype##_kernel, \
        q_proj, k_proj, v_proj, o, past_len, kvbuf, kvidx, layer_idx);                                          \
  }
  ARGS_rotary_mha_decode_kvconst(DISPATCH);
#undef DISPATCH

#define DISPATCH(head_dim, dtype)                                       \
  if (D == head_dim) {                                                  \
    return rotary_mha_decode(                                           \
        launch_rotary_mha_decode_##head_dim##_##dtype##_kernel, q_proj, \
        k_proj, v_proj, o, past_len, kvbuf, kvidx, layer_idx);          \
  }
  ARGS_rotary_mha_decode(DISPATCH);
#undef DISPATCH

  TORCH_CHECK(false, "No suitable kernel. B=", B, " H=", H, " D=", D, " L=", L,
              " maxlen=", maxlen);
}

ITER_rotary_mha_decode_kvconst(DEFINE_rotary_mha_decode_kvconst);
ITER_rotary_mha_decode(DEFINE_rotary_mha_decode);

}  // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ITER_rotary_mha_decode_kvconst(DEFINE_pybind);
  ITER_rotary_mha_decode(DEFINE_pybind);
  m.def("dispatch_rotary_mha_decode", &dispatch_rotary_mha_decode,
        "dispatch_rotary_mha_decode");
}
