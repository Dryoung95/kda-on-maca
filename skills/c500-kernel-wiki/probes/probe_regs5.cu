#include <cstdio>
__global__ void simple(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 1.5f + 0.5f;
}
int main() {
    int n = 1<<20; float* f; cudaMalloc(&f, n*sizeof(float));
    simple<<<(n+255)/256,256>>>(f,n);
    cudaDeviceSynchronize();
    return 0;
}
