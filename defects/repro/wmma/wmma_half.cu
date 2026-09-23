#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// TWO-HALF TEST: the decisive discriminator for the wmma m16n16k16 path.
//
// Lane l in 0..31 (low half)  fills its A-fragment with value 1.
// Lane l in 32..63 (high half) fills its A-fragment with value 2.
// B = 1 everywhere, accumulator = 0.
//
// Then C = A @ B. Every output element is a sum of 16 products, one per lane
// of the warp (because each of the 16 K positions is owned by one lane, and
// with B=1 the sum is over the lanes that own K positions).
//
// What each output element equals:
//   - if the MMA correctly combines BOTH halves : 1*8 + 2*8 = 24  (each lane
//     contributes once; 16 lanes own the 16 K positions)
//   - if the MMA only ever sees the LOW half    : 1*16 = 16
//   - if the MMA only ever sees the HIGH half   : 2*16 = 32
//   - if the hardware silently drops half the lanes: 16 or 32
//
// Note: with a 64-lane warp and only 16 K positions, exactly 16 of the 64
// lanes can own a K position; the other 48 must own something redundant or
// nothing. The value tells us which half the hardware used.
//
// We then vary which half gets 1 vs 2 and compare, to remove any dependence
// on the specific lane->K mapping.
__global__ void k(float* out, int mode) {
    const int lane = __lane_id();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

    // mode 0: low=1, high=2 ; mode 1: low=2, high=1
    const bool low = lane < 32;
    const _Float16 val = (low ? (mode == 0 ? 1.0f : 2.0f)
                              : (mode == 0 ? 2.0f : 1.0f));
    const _Float16 one = 1.0f;
    union { _Float16 v[4]; decltype(a.x) vec; } ua, ub;
    #pragma unroll
    for (int i = 0; i < 4; i++) { ua.v[i] = val; ub.v[i] = one; }
    a.x = ua.vec; b.x = ub.vec;
    wmma::fill_fragment(c, 0.0f);

    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(out, c, 16, wmma::mem_row_major);
}

int main() {
    float* d;
    cudaMallocManaged(&d, 16 * 16 * sizeof(float));
    for (int mode = 0; mode < 2; mode++) {
        k<<<1, 64>>>(d, mode);
        cudaDeviceSynchronize();
        double mn = 1e30, mx = -1e30, sum = 0;
        for (int i = 0; i < 256; i++) {
            mn = fmin(mn, d[i]); mx = fmax(mx, d[i]); sum += d[i];
        }
        printf("mode %d (low=%s high=%s): min=%g max=%g mean=%g\n",
               mode, mode == 0 ? "1" : "2", mode == 0 ? "2" : "1", mn, mx, sum / 256);
    }
    printf("\nInterpretation:\n");
    printf("  16  -> only the LOW half of the warp contributed\n");
    printf("  24  -> both halves contributed correctly\n");
    printf("  32  -> only the HIGH half contributed\n");
    printf("  min==max means every output element saw the same lane set.\n");
    cudaFree(d);
    return 0;
}
