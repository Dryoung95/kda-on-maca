#include <cstdio>
// Force k registers via k independent named accumulators
template<int N>
__global__ void reg_named(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a0=1.0f, a1=2.0f, a2=3.0f, a3=4.0f;
    float acc[N];
    #pragma unroll
    for (int k=0;k<N;k++) acc[k] = a0*(k+1);
    float s = 0;
    #pragma unroll
    for (int k=0;k<N;k++) s += acc[k] * a1 * (float)(i+k);
    if (i < n) out[i] = s;
}
int main() {
    int n = 1<<20; float* f; cudaMalloc(&f, n*sizeof(float));
    reg_named<8><<<(n+255)/256,256>>>(f,n);
    reg_named<32><<<(n+255)/256,256>>>(f,n);
    reg_named<128><<<(n+255)/256,256>>>(f,n);
    cudaDeviceSynchronize(); printf("done\n"); return 0;
}
