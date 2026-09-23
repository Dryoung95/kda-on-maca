#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Decide whether the m16n16k16 wmma path is internally consistent WITHOUT
// any load_matrix_sync. Every fragment is filled with a KNOWN scalar value,
// so there is no load, no memory addressing, and no fragment copying.
//
// Lane l of the warp holds A-fragment value  = 1000 + l
//                    B-fragment value  = 1
//                    accumulator        = 0
// C = A * B computed by mma_sync, stored with mem_row_major.
//
// We then read back the matrix. If the mma + store path is self-consistent,
// the stored value at (r,c) tells us exactly which lane contributed there.
//
// What the header store does for accumulator, 16x16x16, mem_row_major:
//     row = (lane >> 4) << 2;  col = lane & 0xf;
//     p[row*ldm+col]     = f.x[0];
//     p[(row+1)*ldm+col] = f.x[1];
//     p[(row+2)*ldm+col] = f.x[2];
//     p[(row+3)*ldm+col] = f.x[3];
// So element index i within a lane lands at (row+i, col).
//
// A correct MMA on NVIDIA hardware: with A uniform-per-lane = (1000+lane)
// and B = 1, output(r,c) = sum over k of A(r,k)*B(k,c) = 16*(1000+lane_of(r,c))
// where lane_of is the lane owning output element (r,c) = lane owning the
// 4-row group containing r, at column c.
//
// We just print the matrix and see which lanes' values show up where.
__global__ void k(float* out) {
    const int lane = __lane_id();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

    // uniform scalar fill. .x is a vector type, so write through a union.
    const _Float16 hv = (_Float16)(1000 + lane);
    const _Float16 one = (_Float16)1.0f;
    union { _Float16 v[4]; decltype(a.x) vec; } ua, ub;
    #pragma unroll
    for (int i = 0; i < 4; i++) { ua.v[i] = hv; ub.v[i] = one; }
    a.x = ua.vec;
    b.x = ub.vec;
    wmma::fill_fragment(c, 0.0f);

    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(out, c, 16, wmma::mem_row_major);
}

int main() {
    float* d;
    cudaMallocManaged(&d, 16 * 16 * sizeof(float));
    k<<<1, 64>>>(d);
    cudaDeviceSynchronize();
    printf("C = A @ B, A fragment on lane l = 1000+l (all 4 elems), B = 1\n");
    printf("Each output value is 16*(1000+lane). Decode: (value/16 - 1000) = lane.\n\n");
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            int lane = (int)(d[r * 16 + c] / 16.0f + 0.5f) - 1000;
            printf("%4d ", lane);
        }
        printf("  <- row %d\n", r);
    }
    printf("\nIf the mma+store path were correct, each row r would decode to the\n");
    printf("SAME lane across the whole row, and lanes 0..63 would each appear\n");
    printf("exactly 4 times (4 elements x 16 columns = 64 outputs per lane group).\n");
    cudaFree(d);
    return 0;
}
