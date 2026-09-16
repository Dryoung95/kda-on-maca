#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void k_cg(int* out) {
    cg::grid_group g = cg::this_grid();
    if (g.is_valid()) {
        g.sync();
        if (threadIdx.x == 0) out[blockIdx.x] = (int)g.size();
    } else {
        if (threadIdx.x == 0) out[blockIdx.x] = -1;
    }
}
int main() {
    int* i; cudaMalloc(&i, 1024*sizeof(int));
    cudaMemset(i, 0, 1024*sizeof(int));
    k_cg<<<8,64>>>(i);
    cudaError_t e = cudaDeviceSynchronize();
    printf("grid_group.sync(): %s\n", cudaGetErrorString(e));
    // read back
    int host[8]; cudaMemcpy(host, i, 8*sizeof(int), cudaMemcpyDeviceToHost);
    printf("  out[0]=%d\n", host[0]);
    return 0;
}
