#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float* Q,const float* K,const float* V,const int N,const int d,const int Tc,const int Tr, const int Bc,const int Br,const float scale,float* l,float* m,float* O){
    int tid = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    extern __shared__ float sram[];
    int tile_size = Bc*d;
    float* Qi = sram;
    float* Ki = &sram[tile_size];
    float* Vi = &sram[tile_size*2];
    float* S = &sram[tile_size*3];
    
}