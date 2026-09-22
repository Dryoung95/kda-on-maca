#include <cstdio>
#include <cuda_runtime.h>
#include <mctlass/mctlass.h>
#include <mctlass/gemm/device/gemm.h>
int main(){
  for(int n : {64,96,128,192,256}){
    float*A,*B,*C;
    cudaMallocManaged(&A,4*n*n);cudaMallocManaged(&B,4*n*n);cudaMallocManaged(&C,4*n*n);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){A[i*n+j]=(i==j)?1.f:0.f;B[i*n+j]=(i==j)?1.f:0.f;C[i*n+j]=0.f;}
    using Gemm=mctlass::gemm::device::Gemm<float,mctlass::layout::RowMajor,
      float,mctlass::layout::RowMajor,float,mctlass::layout::RowMajor>;
    Gemm g; float al=1.f,be=0.f;
    Gemm::Arguments args{{n,n,n},{A,n},{B,n},{C,n},{C,n},{al,be},1};
    size_t ws=Gemm::get_workspace_size(args); void* w=nullptr;
    if(ws>0)cudaMalloc(&w,ws);
    g.initialize(args,w); g(); cudaDeviceSynchronize(); if(w)cudaFree(w);
    int bad=0,firstBad=-1,lastBad=-1,badPerBadRow=0,badRows=0;
    for(int i=0;i<n;i++){int b=0;for(int j=0;j<n;j++)if(C[i*n+j]!=A[i*n+j])b++;
      if(b>0){bad+=b;badRows++;if(firstBad<0)firstBad=i;lastBad=i;badPerBadRow=b;}}
    fprintf(stderr,"n=%3d totalBad=%4d badRows=%3d first=%3d last=%3d elemsPerBadRow=%d\n",n,bad,badRows,firstBad,lastBad,badPerBadRow);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  return 0;
}
