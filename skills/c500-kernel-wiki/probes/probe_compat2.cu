#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k_vec(float* out) {
    int t = threadIdx.x;
    float4 v = {t*1.f, t*2.f, t*3.f, t*4.f};
    float4 w = {1.f, 1.f, 1.f, 1.f};
    float4 r;
    r.x = v.x*w.x; r.y = v.y*w.y; r.z = v.z*w.z; r.w = v.w*w.w;
    if (r.x < -1e30f) out[blockIdx.x] = r.x;
}

__global__ void k_half(__half* out) {
    int t = threadIdx.x;
    __half a = __float2half((float)t), b = __float2half(2.0f);
    __half r = __hadd(a, b);
    if (r < __float2half(-1e30f)) out[blockIdx.x] = r;
}

__global__ void k_math(float* out) {
    int t = threadIdx.x;
    float x = (float)t * 0.001f;
    float r = __expf(x) + __logf(x+1.0f) + __sinf(x) + sqrtf(x);
    if (r < -1e30f) out[blockIdx.x] = r;
}

__launch_bounds__(64, 16)
__global__ void k_launchbounds(int* out) {
    int t = threadIdx.x;
    int v = t;
    for (int k = 0; k < 8; k++) v = v * (k+1) + k;
    if (v == 0xdeadbeef) out[blockIdx.x] = v;
}

__global__ void k_cg(int* out) {

    cg::grid_group g = cg::this_grid();
    g.sync();
    if (threadIdx.x == 0) out[blockIdx.x] = g.size();
}

int main() {
    float* f; int* i; __half* h;
    cudaMalloc(&f, 1024*sizeof(float)); cudaMalloc(&i, 1024*sizeof(int)); cudaMalloc(&h, 1024*sizeof(__half));
    k_vec<<<8,64>>>(f);        printf("float4 vec ops    : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_half<<<8,64>>>(h);       printf("__half arithmetic : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_math<<<8,64>>>(f);       printf("fast math builtins: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_launchbounds<<<8,64>>>(i); printf("__launch_bounds__ : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_cg<<<8,64>>>(i);         printf("cooperative_groups: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
