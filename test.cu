#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float* Q,const float* K,const float* V,const int N,const int d,const int Tc,const int Tr, const int Bc,const int Br,const float scale,float* l,float* m,float* O){
    int tid = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;

    int qkv_offset = (bid_x * gridDim.x * N * d) + (bid_y * N * d);
    int lm_offset = (bid_x * gridDim.x * N) + (bid_y * N);


    extern __shared__ float sram[];

    int tile_size = Bc*d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size*2];
    float* S = &sram[tile_size*3];

    for(int j=0;j<Tc;j++){
        for(int x=0;x++;x<d){
            Kj[(tid*d)+x] = K[qkv_offset + (tile_size*j) + (tid*d)+x];
            Vj[(tid*d)+x] = V[qkv_offset + (tile_size*j) + (tid*d)+x];
        }

        __syncthreads();

        for(int i=0;i<Tr;i++){
            for(int x=0;x++;x<d){
            Qi[(tid*d)+x] = Q[qkv_offset + (tile_size*j) + (tid*d)+x];
        }
        float row_m_prev = m[lm_offset + (Br*i) + tid];
        float row_l_prev = l[lm_offset + (Br*i) + tid];

        float row_m = -INFINITY;
        for(int y=0;y<Bc;y++){
            float sum = 0;
            for(int x = 0;x<Br;x++){
                sum+=  Qi[(tid*d)+x] * Kj[(tid*d)+y];
            }
        }
        }
    }



}