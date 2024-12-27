#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float* Q,const float* K,const float* V,const int N,const int d,const int Tc,const int Tr, const int Bc,const int Br,const float scale,float* l,float* m,float* O){
    
}