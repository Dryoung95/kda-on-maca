#include <cstdio>

template<int N>
__global__ void reg_heavy(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v[N];
    #pragma unroll
    for (int k = 0; k < N; k++) v[k] = (float)k * 0.001f;
    if (i < n) {
        float acc = 0;
        #pragma unroll
        for (int k = 0; k < N; k++) acc += v[k] * (float)(i + k);
        out[i] = acc;
    }
}

int main() {
    int n = 1<<20;
    float* f; cudaMalloc(&f, n*sizeof(float));
    printf("=== register pressure sweep (block=256, 1M elements) ===\n");
    // dispatch by N at runtime via macro expansion
    #define RUN(N) do { \
        for (int w=0;w<5;w++) reg_heavy<N><<<(n+255)/256, 256>>>(f, n); \
        cudaDeviceSynchronize(); \
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); \
        cudaEventRecord(a); \
        for (int r=0;r<50;r++) reg_heavy<N><<<(n+255)/256, 256>>>(f, n); \
        cudaEventRecord(b); cudaDeviceSynchronize(); \
        float ms=0; cudaEventElapsedTime(&ms,a,b); \
        printf("  unroll=%3d  %8.2f us\n", N, ms*1000/50); \
        cudaEventDestroy(a); cudaEventDestroy(b); \
    } while(0)
    RUN(4); RUN(8); RUN(16); RUN(32); RUN(64); RUN(96); RUN(128);
    return 0;
}
