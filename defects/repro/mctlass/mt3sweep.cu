#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>

// RowMajor primary operator (per the memory note, the ColumnMajor output
// specialization is the one that mis-transposes). Test both.
using RowGemm = mctlass::gemm::device::Gemm<
    float, mctlass::layout::RowMajor,
    float, mctlass::layout::RowMajor,
    float, mctlass::layout::RowMajor>;

static void run(const char* tag, int m, int n, int k) {
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float)*m*k);
    cudaMallocManaged(&B, sizeof(float)*k*n);
    cudaMallocManaged(&C, sizeof(float)*m*n);
    // A = identity-ish pattern, B = identity
    for (int i = 0; i < m*k; i++) A[i] = 0.0f;
    for (int i = 0; i < m && i < k; i++) A[i*k+i] = 1.0f;
    for (int i = 0; i < k*n; i++) B[i] = 0.0f;
    for (int i = 0; i < k && i < n; i++) B[i*n+i] = 1.0f;
    for (int i = 0; i < m*n; i++) C[i] = 0.0f;

    RowGemm gemm_op;
    float alpha = 1.0f, beta = 0.0f;
    RowGemm::Arguments args{
        {m, n, k},
        {A, k}, {B, n}, {C, n}, {C, n},
        {alpha, beta}, 1
    };
    size_t ws = RowGemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    mctlass::Status st = gemm_op.initialize(args, workspace);
    if (st != mctlass::Status::kSuccess) { fprintf(stderr, "%s: init FAILED\n", tag); return; }
    st = gemm_op();
    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);

    int bad = 0, firstBad = -1;
    for (int i = 0; i < m*n; i++) {
        float want = (i % n == i / n && (i/n) < m && (i%n) < k && (i/n) < n) ? 1.0f : 0.0f;
        // C[i*n+j] should be 1 where i==j within min(m,n,k)
        int r = i / n, c = i % n;
        float want2 = (r == c && r < k) ? 1.0f : 0.0f;
        if (C[i] != want2) { bad++; if (firstBad < 0) firstBad = i; }
    }
    fprintf(stderr, "%-22s m=%d n=%d k=%d mismatched=%d (of %d) firstBad=%d (r=%d c=%d got=%.3f)\n",
            tag, m, n, k, bad, m*n, firstBad,
            firstBad>=0?firstBad/n:-1, firstBad>=0?firstBad%n:-1,
            firstBad>=0?C[firstBad]:0);
    cudaFree(A); cudaFree(B); cudaFree(C);
}

int main() {
    run("square 128",   128, 128, 128);
    run("square 64",     64,  64,  64);
    run("square 256",   256, 256, 256);
    run("nonsquare A",  128,  64,  96);
    run("nonsquare B",   64, 128,  96);
    run("tall",         200,  64,  64);
    return 0;
}
