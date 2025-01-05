#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            throw std::runtime_error("CUDA call failed");                         \
        }                                                                         \
    } while (0)

__global__ void forward_kernel_v1(const float* Q,const float* K,const float* V,const int N,const int d,const int Tc,const int Tr, const int Bc,const int Br,const float scale,float* l,float* m,float* O){
    int tid = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;

    int qkv_offset = (bid_x * gridDim.y * N * d) + (bid_y * N * d);
    int lm_offset = (bid_x * gridDim.y * N) + (bid_y * N);


    extern __shared__ float sram[];

    int tile_size = Bc*d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size*2];
    float* S = &sram[tile_size*3];

    for(int j=0;j<Tc;j++){
        for(int x=0;x<d;x++){
            Kj[(tid*d)+x] = K[qkv_offset + (tile_size*j) + (tid*d)+x];
            Vj[(tid*d)+x] = V[qkv_offset + (tile_size*j) + (tid*d)+x];
        }

        __syncthreads();

        for(int i=0;i<Tr;i++){
            for(int x=0;x<d;x++){
            Qi[(tid*d)+x] = Q[qkv_offset + (tile_size*i) + (tid*d)+x];
        }
        float row_m_prev = m[lm_offset + (Br*i) + tid];
        float row_l_prev = l[lm_offset + (Br*i) + tid];

        float row_m = -INFINITY;
        for(int y=0;y<Bc;y++){
            float sum = 0;
            for(int x = 0;x<d;x++){
                sum+=  Qi[(tid*d)+x] * Kj[(y*d)+x];
            }
            sum*=scale;
            S[(Bc*tid) + y] = sum;
            if(sum>row_m){
                row_m = sum;
                }
            }
        float row_l = 0;
        for(int y =0;y<Bc;y++){
            S[(Bc * tid) + y] = __expf(S[(Bc * tid) + y] - row_m);
            row_l +=  S[(Bc * tid) + y];
            }
        
        float row_m_new = max(row_m_prev,row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        for(int x = 0; x < d;x++){
                float pv = 0;
                for(int y=0;y<Bc;y++){
                    pv += S[(Bc * tid) + y] * Vj[(y*d)+x];
                }
                O[qkv_offset + (tile_size*i) + (tid*d)+x] = (1/(row_l_new)) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size*i) + (tid*d)+x]) + (__expf(row_m - row_m_new)*pv));
            }
            m[lm_offset + (Br*i) + tid] = row_m_new;
            l[lm_offset + (Br*i) + tid] = row_l_new;
        }

        __syncthreads();
    }

}


void forward(float* Q, float* K, float* V, float* O, int B, int nh, int N, int d) {
    const int Bc = 32;
    const int Br = 32;

    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;

    const float scale = 1.0f / sqrtf(d);

    int sram_size = (4 * Bc * d + Bc * Br) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    std::cout << "Max shared memory: " << max_sram_size << ", requested shared memory: " << sram_size << std::endl;

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel_v1<<<grid_dim, block_dim, sram_size>>>(Q, K, V, N, d, Tc, Tr, Bc, Br, scale, O);
    cudaDeviceSynchronize();
}
