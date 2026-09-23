import sys, os, torch
from torch.utils.cpp_extension import load_inline

SRC = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// 256 blocks. block b = (lane b/4, elem b%4). Set exactly that one element.
__global__ void map_kernel(float* out, int col) {
    const int b = blockIdx.x;
    const int lane = b / 4;
    const int elem = b % 4;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
    #pragma unroll
    for (int i = 0; i < f.num_elements; i++) f.x[i] = 0.0f;
    if (threadIdx.x == lane) f.x[elem] = 1.0f;
    wmma::store_matrix_sync(out + b * 256, f, 16,
        col ? wmma::mem_col_major : wmma::mem_row_major);
}

torch::Tensor fmap(int col) {
    auto out = torch::zeros({256, 16, 16}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    map_kernel<<<256, 64>>>(out.data_ptr<float>(), col);
    return out;
}
"""
os.makedirs('/tmp/t/diagbuild/fmap2', exist_ok=True)
m = load_inline(name='fmap2', cpp_sources=["torch::Tensor fmap(int col);"],
    cuda_sources=[SRC], functions=['fmap'],
    build_directory='/tmp/t/diagbuild/fmap2', verbose=False)

n = 16
res = {}
for col in (0, 1):
    out = m.fmap(col).cpu()
    res[col] = {}
    for b in range(256):
        nz = (out[b] == 1.0).nonzero()
        res[col][b] = (int(nz[0,0]), int(nz[0,1])) if nz.numel() else None

def mrow(lane, e):
    v = res[0].get(lane*4+e)
    return f"({v[0]:2d},{v[1]:2d})" if v else "  --  "

def mcol(lane, e):
    v = res[1].get(lane*4+e)
    return f"({v[0]:2d},{v[1]:2d})" if v else "  --  "

print("TRUE fragment->memory map (row, col)")
print("lane  | mem_row_major elem0 elem1 elem2 elem3  || mem_col_major elem0 elem1 elem2 elem3")
for lane in range(64):
    r = "  ".join(mrow(lane, e) for e in range(4))
    c = "  ".join(mcol(lane, e) for e in range(4))
    print(f"{lane:5d} |  {r}  ||  {c}")

# Are the two outputs exact transposes of each other?
pairs_ok = 0
for b in range(256):
    v0, v1 = res[0][b], res[1][b]
    if v0 and v1 and v0[0] == v1[1] and v0[1] == v1[0]:
        pairs_ok += 1
print(f"\n(lane,elem) pairs where row_major and col_major outputs are exact transposes: {pairs_ok}/256")

# Is col_major's output the transpose of what row_major produces for the
# TRANSPOSED logical (row,col)? i.e. does the tag actually do anything at all?
same = sum(1 for b in range(256) if res[0][b] == res[1][b])
print(f"(lane,elem) pairs where both tags give the SAME output: {same}/256")
