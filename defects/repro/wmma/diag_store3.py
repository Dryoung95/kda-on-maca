import sys, os, torch
from torch.utils.cpp_extension import load_inline

# Definitive test of D2 ("store_matrix_sync ignores layout tag").
# The original repro (diag_store2.py) casts INTEGER CONSTANTS 16 and 17 to
# wmma::layout_t. But the MACA enum is:
#     enum layout_t { mem_row_major, mem_col_major };
# i.e. mem_row_major == 0 and mem_col_major == 1. Values 16/17 are NOT valid
# enumerators. Anything out of range is undefined behaviour and landing in the
# same branch for both constants would look exactly like "tag ignored".
#
# Here we pass the real enumerators via template parameters 0 and 1, with no
# cast at all.

SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// The fragment value is set so that the CORRECT row-major output is
// out[r][c] = 100*r + c, i.e. it encodes (row, col) directly.
template<int LAYOUT>
__global__ void store_kernel(float* C, int n) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        f.x[i] = 100.0f * ((__lane_id() >> 4) * 4 + i) + (__lane_id() & 0xf);
    wmma::store_matrix_sync(C, f, (unsigned)n, (wmma::layout_t)LAYOUT);
}

torch::Tensor store(int layout) {
    auto C = torch::zeros({16,16}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    if (layout == 0) store_kernel<0><<<1,64>>>(C.data_ptr<float>(), 16);
    else             store_kernel<1><<<1,64>>>(C.data_ptr<float>(), 16);
    return C;
}
"""
os.makedirs('/tmp/t/diagbuild/store3', exist_ok=True)
m = load_inline(name='diag_store3',
    cpp_sources=["torch::Tensor store(int layout);"],
    cuda_sources=[SRC], functions=['store'],
    build_directory='/tmp/t/diagbuild/store3', verbose=False)

n = 16
expected_row = torch.arange(n*n).float().reshape(n,n)          # out[r][c] = 100r+c
expected_col = expected_row.T.contiguous()

for layout, name, expected in [(0, "mem_row_major (real enum 0)", expected_row),
                               (1, "mem_col_major (real enum 1)", expected_col)]:
    out = m.store(layout)
    torch.cuda.synchronize()
    ok = torch.equal(out.cpu().to(torch.int64), expected.to(torch.int64))
    print(f"{name}: matches_expected={ok}")
    if not ok:
        print("  out[0,:4] =", out[0,:4].cpu().numpy().astype(int),
              " expected", expected[0,:4].numpy().astype(int))
        print("  out[:,0]  =", out[:,0].cpu().numpy().astype(int),
              " expected", expected[:,0].numpy().astype(int))

# Are the two outputs actually different from each other?
a = m.store(0).cpu()
b = m.store(1).cpu()
print("\nrow_major output equals col_major output:", bool(torch.equal(a, b)))
print("col_major output equals transpose of row_major output:", bool(torch.equal(b, a.T.contiguous())))
