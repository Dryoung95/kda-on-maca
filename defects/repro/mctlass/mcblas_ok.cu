#include <cstdio>
#include <cuda_runtime.h>
#include <mcblas/mcblas.h>

// Verify mcblas (the path the report recommends as correct) really is correct,
// so the "avoid mctlass/wmma, use mcblas" guidance is sound.
int main() {
    const int N = 256;
    float *A, *B, *C;
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        A[i*N+j] = (i==j) ? 1.0f : 0.0f;
        B[i*N+j] = (i==j) ? 2.0f : 0.0f;
        C[i*N+j] = 0.0f;
    }
    mcblasHandle_t h;
    mcblasStatus_t st = mcblasCreate(&h);
    if (st != MCBLAS_STATUS_SUCCESS) { printf("mcblasCreate failed %d\n", (int)st); return 1; }
    float a = 1.0f, b = 0.0f;
    st = mcblasSgemm(h, MCBLAS_OP_N, MCBLAS_OP_N, N, N, N, &a,
                     A, N, B, N, &b, C, N);
    cudaDeviceSynchronize();
    if (st != MCBLAS_STATUS_SUCCESS) { printf("mcblasSgemm failed %d\n", (int)st); return 1; }
    int bad = 0;
    for (int i = 0; i < N*N; i++) {
        float want = (i/N == i%N) ? 2.0f : 0.0f;
        if (C[i] != want) bad++;
    }
    printf("mcblasSgemm N=%d mismatched=%d (expect 0)\n", N, bad);
    mcblasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
