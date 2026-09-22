#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>

void run_mctlass(float* A, float* B, float* C, int n) {
    using Gemm = mctlass::gemm::device::Gemm<
        float, mctlass::layout::ColumnMajor,
        float, mctlass::layout::ColumnMajor,
        float, mctlass::layout::ColumnMajor>;
    Gemm gemm_op;
    float alpha = 1.0f, beta = 0.0f;
    Gemm::Arguments args{
        {n, n, n},
        {B, n},
        {A, n},
        {C, n},
        {C, n},
        {alpha, beta},
        1
    };
    size_t ws = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    mctlass::Status status = gemm_op.initialize(args, workspace);
    fprintf(stderr, "init=%d ws=%zu\n", (int)status, ws);
    if (status != mctlass::Status::kSuccess) { if (workspace) cudaFree(workspace); return; }
    status = gemm_op();
    fprintf(stderr, "run=%d\n", (int)status);
    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);
}

int main() {
    int n = 128;
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float)*n*n);
    cudaMallocManaged(&B, sizeof(float)*n*n);
    cudaMallocManaged(&C, sizeof(float)*n*n);
    // A = simple pattern, B = identity
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        A[i*n+j] = (i==j) ? 1.0f : 0.0f;
        B[i*n+j] = (i==j) ? 1.0f : 0.0f;
        C[i*n+j] = 0.f;
    }
    run_mctlass(A, B, C, n);
    // C should equal A (identity * identity)
    int bad = 0;
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++)
        fprintf(stderr, "%7.3f ", C[i*n+j]);
      fprintf(stderr, "\n");
    }
    for (int i = 0; i < n*n; i++) if (C[i] != A[i]) bad++;
    fprintf(stderr, "n=%d mismatched=%d (expect 0)\n", n, bad);
    return 0;
}
