#include <cstdio>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>
using Gemm = mctlass::gemm::device::Gemm<
    float, mctlass::layout::ColumnMajor,
    float, mctlass::layout::ColumnMajor,
    float, mctlass::layout::ColumnMajor>;
int main() {
    // sizeof probes: which sub-object is huge?
    fprintf(stderr, "sizeof(Params)=%zu sizeof(SharedStorage)=%zu\n",
        sizeof(Gemm::GemmKernel::Params), sizeof(Gemm::GemmKernel::SharedStorage));
    // stack-allocate a Gemm and confirm
    Gemm* p = new Gemm();
    fprintf(stderr, "constructed via new\n");
    delete p;
    return 0;
}
