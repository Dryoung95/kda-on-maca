#include <cstdio>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>

using RowGemm = mctlass::gemm::device::Gemm<
    float, mctlass::layout::RowMajor,
    float, mctlass::layout::RowMajor,
    float, mctlass::layout::RowMajor>;

static void run(const char* tag, int m, int n, int k) {
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float)*m*k);
    cudaMallocManaged(&B, sizeof(float)*k*n);
    cudaMallocManaged(&C, sizeof(float)*m*n);
    for (int i = 0; i < m*k; i++) A[i] = 0.0f;
    for (int i = 0; i < m && i < k; i++) A[i*k+i] = 1.0f;
    for (int i = 0; i < k*n; i++) B[i] = 0.0f;
    for (int i = 0; i < k && i < n; i++) B[i*n+i] = 1.0f;
    for (int i = 0; i < m*n; i++) C[i] = 0.0f;

    RowGemm gemm_op;
    RowGemm::Arguments args{{m,n,k},{A,k},{B,n},{C,n},{C,n},{1.0f,0.0f},1};
    size_t ws = RowGemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    gemm_op.initialize(args, workspace);
    gemm_op();
    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);

    fprintf(stderr, "%s  m=%d n=%d k=%d  bad rows (r=bad):", tag, m, n, k);
    for (int r = 0; r < m; r++) {
        bool rowok = true;
        for (int c = 0; c < n; c++)
            if (C[r*n+c] != (r == c && r < k ? 1.0f : 0.0f)) { rowok = false; break; }
        if (!rowok) fprintf(stderr, " %d", r);
    }
    fprintf(stderr, "\n");
    // r%16 histogram of bad rows
    int hist[16] = {0};
    for (int r = 0; r < m; r++) {
        bool rowok = true;
        for (int c = 0; c < n; c++)
            if (C[r*n+c] != (r == c && r < k ? 1.0f : 0.0f)) { rowok = false; break; }
        if (!rowok) hist[r % 16]++;
    }
    fprintf(stderr, "   bad rows by r%%16: [");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%d%s", hist[i], i<15?" ":"");
    fprintf(stderr, "]\n");
    cudaFree(A); cudaFree(B); cudaFree(C);
}

int main() {
    run("sq64", 64, 64, 64);
    run("sq128", 128, 128, 128);
    return 0;
}
