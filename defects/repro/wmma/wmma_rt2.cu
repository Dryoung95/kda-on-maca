#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Round-trip test with a value encoding that makes the defect unambiguous.
//
// src[r][c] = 256*r + c  (unique per element, in [0, 65520]).
// After load_matrix_sync(A, src) we expect, IF load is correct, that lane l
// holds the 4 elements
//     (l&0xf, col),  col = ((l>>4)<<2) + i,  i = 0..3
// i.e. lane l owns row (l&0xf) and 4 consecutive columns starting at
// ((l>>4)<<2). Note the header's A row_major load uses
//     row = lane & 0xf;  col = (lane >> 4) << 2
// which is exactly this. So the LANE->ADDRESS mapping is provably correct.
//
// The question is whether the data actually ARRIVES in the right fragment
// element. We test by mma against identity in the K dimension:
//     B = identity in memory, loaded as matrix_b col_major.
// Then C(r,c) = sum_k A(r,k) * B(k,c) = A(r,c). So C should equal src.
//
// If load is correct: C == src exactly (256/256).
// If the fragment data is shuffled/misrouted, C differs and the pattern
// reveals which lanes got wrong data.
__global__ void rt(float* C, const __half* src) {
    __shared__ __half sA[16][16];
    __shared__ __half sB[16][16];
    const int tid = threadIdx.x;
    if (tid < 256) {
        int r = tid / 16, c = tid % 16;
        sA[r][c] = src[r * 16 + c];
        sB[r][c] = (r == c) ? __float2half(1.0f) : __float2half(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

    wmma::load_matrix_sync(a, &sA[0][0], 16);
    wmma::load_matrix_sync(b, &sB[0][0], 16);
    wmma::fill_fragment(c, 0.0f);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    __half* h;
    float* C;
    cudaMallocManaged(&h, 256 * sizeof(__half));
    cudaMallocManaged(&C, 256 * sizeof(float));
    for (int i = 0; i < 256; i++) {
        int r = i / 16, c = i % 16;
        h[i] = __float2half((float)(256 * r + c));
    }
    rt<<<1, 64>>>(C, h);
    cudaDeviceSynchronize();

    int match = 0;
    // For each lane l, the header says element i lands at (row=(l&0xf)+i? no...)
    // store: row = (l>>4)<<2, col = l&0xf; element i -> row+i, col.
    // so output (r,c) is written by lane l = ((r>>2)<<4) + c, element r&3.
    printf("row | first 8 output values (expected 256r+c) | lanes that should own them\n");
    for (int r = 0; r < 16; r++) {
        printf("%3d | ", r);
        for (int c = 0; c < 8; c++) {
            float want = 256 * r + c;
            float got = C[r * 16 + c];
            if (got == want) match++;
            printf("%6.0f", got);
        }
        printf("   lanes ");
        for (int c = 0; c < 4; c++) printf("%d ", ((r >> 2) << 4) + c);
        printf("\n");
    }
    printf("\nexact element matches: %d/256\n", match);
    cudaFree(h); cudaFree(C);
    return 0;
}
