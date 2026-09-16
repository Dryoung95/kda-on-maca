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
    reg_heavy<16><<<(n+255)/256, 256>>>(f, n);
    reg_heavy<64><<<(n+255)/256, 256>>>(f, n);
    reg_heavy<128><<<(n+255)/256, 256>>>(f, n);
    cudaDeviceSynchronize();
    return 0;
}
