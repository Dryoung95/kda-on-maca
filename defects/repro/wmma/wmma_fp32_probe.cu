#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
__global__ void k(float* C){
  // if fp32 input fragments are unsupported this file must not compile
  wmma::fragment<wmma::matrix_a, 16,16,16, float, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16,16,16, float, wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16, float> acc;
  wmma::fill_fragment(a, 1.0f);
  wmma::fill_fragment(b, 1.0f);
  wmma::fill_fragment(acc, 0.0f);
  wmma::mma_sync(acc, a, b, acc);
  wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
}
int main(){ float*C; cudaMallocManaged(&C,4*256); k<<<1,64>>>(C); cudaDeviceSynchronize();
  printf("fp32-fragment wmma built and ran; C[0]=%.1f C[255]=%.1f\n",C[0],C[255]); return 0; }
