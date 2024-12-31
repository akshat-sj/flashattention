#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel_basic(const float* Q,const float* K,const float* V,const int N,const int d,const int Tc,const int Tr, const int Bc,const int Br,const float scale,float* l,float* m,float* O){
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

template <int Br, int Bc, int d>
__global__ void forward_kernel(
    const float* Q, const float* K, const float* V, 
    const int N, const int d, const int Tc, const int Tr, 
    const int Bc, const int Br,const int batch_stride, const float scale, 
    float* l, float* m, float* O) {
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int batch_offset = blockIdx.x * batch_stride; // Batch-specific offset

    __shared__ float Qi[Br][d]; 
    __shared__ float Kj[Br][Bc];
    __shared__ float Vj[Br][d];

    __shared__ float Si[Br][Bc];

    const int num_tiles = d/Bc; 

    float Oi[num_tiles];
    float li = 0.0f;
    float mi = - INFINITY;
    for(int x = 0;x<num_tiles;x++){
        Oi[x] = 0;
        Qi[tid_y][x*Bc + tid_x] = Q[batch_offset + (blockIdx.y * Br + tid_y)*d +(x*Bc + tid_x)];
    }
    __syncthreads();

    for(int j=0;j<Tc;j++){
        
    }



}

torch::Tensor forward(torch::Tensor Q,torch::Tensor K, torch::Tensor V){
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float) N/ Bc);
    const int Tr = ceil((float) N/ Br);

    const float scale  = 1.0/sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B,nh,N});
    auto m = torch::full({B,nh,N},-INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (3*Bc*d + Bc*Br) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B,nh);
    dim3 block_dim(Bc);

    forward_kernel_basic<<<grid_dim,block_dim,sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());

return O;

}