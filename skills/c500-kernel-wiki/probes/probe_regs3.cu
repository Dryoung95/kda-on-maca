#include <cstdio>
// volatile array forces material registers, prevents DCE
template<int N>
__global__ void reg_real(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    volatile float v[N];
    for (int k = 0; k < N; k++) v[k] = (float)k * 0.001f;
    float acc = 0;
    for (int k = 0; k < N; k++) { float t = v[k]; acc += t * (float)(i + k); }
    if (i < n) out[i] = acc;
}
int main() {
    int n = 1<<20;
    float* f; cudaMalloc(&f, n*sizeof(float));
    reg_real<16><<<(n+255)/256, 256>>>(f, n);
    reg_real<48><<<(n+255)/256, 256>>>(f, n);
    reg_real<96><<<(n+255)/256, 256>>>(f, n);
    reg_real<160><<<(n+255)/256, 256>>>(f, n);
    cudaDeviceSynchronize();
    printf("done\n");
    return 0;
}
