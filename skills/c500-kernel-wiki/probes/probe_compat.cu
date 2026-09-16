#include <cstdio>

__global__ void k_shfl(float* out) {
    int lane = threadIdx.x;
    float v = (float)lane;
    // full 64-bit mask
    float r = __shfl_sync(0xffffffffffffffffULL, v, 0);
    // warp reduce via derived depth
    for (int off = warpSize/2; off > 0; off >>= 1) r += __shfl_down_sync(0xffffffffffffffffULL, v, off);
    if (lane == 0) out[blockIdx.x] = r;
}

__global__ void k_ballot(int* out) {
    bool cond = (threadIdx.x & 1) == 0;
    unsigned long long mask = __ballot_sync(0xffffffffffffffffULL, cond);
    if (threadIdx.x == 0) out[blockIdx.x] = (int)(mask & 0xffffffff);
}

__global__ void k_shmem_dyn(float* out, int n) {
    extern __shared__ float smem[];
    int t = threadIdx.x;
    smem[t] = (float)t;
    __syncthreads();
    float v = 0;
    for (int off = warpSize/2; off > 0; off >>= 1) {}
    if (t == 0) out[blockIdx.x] = smem[n-1];
}

__global__ void k_fma(float* out) {
    int t = threadIdx.x;
    float a=t*0.5f, b=t*1.5f, c=t*2.5f;
    float r = __fmaf_rn(a,b,c);
    if (r < -1e30f) out[blockIdx.x] = r;
}

__global__ void k_atomic(int* out) {
    int t = threadIdx.x;
    atomicAdd(&out[0], t);
}

int main() {
    float* f; int* i;
    cudaMalloc(&f, 1024*sizeof(float)); cudaMalloc(&i, 1024*sizeof(int));
    cudaMemset(i, 0, 1024*sizeof(int));
    k_shfl<<<8,64>>>(f);                    printf("shfl(64-mask)        : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_ballot<<<8,64>>>(i);                  printf("ballot(64-mask)      : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_shmem_dyn<<<8,64,64*sizeof(float)>>>(f, 64); printf("dynamic shmem       : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_fma<<<8,64>>>(f);                     printf("__fmaf_rn            : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    k_atomic<<<8,64>>>(i);                  printf("atomicAdd            : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
