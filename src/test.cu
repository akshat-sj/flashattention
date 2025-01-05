#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

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



__global__ void forward_kernel_v2(const float* Q, const float* K, const float* V, int N, int d, int Tc, int Tr, int Bc, int Br, float scale, float* O, float* L) {
    int tid = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;

    int qkv_offset = (bid_x * gridDim.y * N * d) + (bid_y * N * d);

    extern __shared__ float sram[];

    int tile_size = Bc * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[2 * tile_size];
    float* S = &sram[3 * tile_size];

    for (int i = 0; i < Tr; i++) {
        for (int x = 0; x < d; x++) {
            Qi[(tid * d) + x] = Q[qkv_offset + (tile_size * i) + (tid * d) + x];
        }

        __syncthreads();

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        float row_m_new ;
        float row_l_new;

        for (int j = 0; j < Tc; j++) {
            for (int x = 0; x < d; x++) {
                Kj[(tid * d) + x] = K[qkv_offset + (tile_size * j) + (tid * d) + x];
                Vj[(tid * d) + x] = V[qkv_offset + (tile_size * j) + (tid * d) + x];
            }

            float row_m = -INFINITY;

            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tid * d) + x] * Kj[(y * d) + x];
                }
                sum *= scale;
                S[(Bc * tid) + y] = sum;
                if (sum > row_m) {
                    row_m = sum;
                }
            }

            row_m_new = max(row_m_prev, row_m);

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tid) + y] = __expf(S[(Bc * tid) + y] - row_m_new);
                row_l += S[(Bc * tid) + y];
            }

            row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tid) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tid * d) + x] = (__expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tid * d) + x] + pv);
            }

            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }

        __syncthreads();

        for (int x = 0; x < d; x++) {
            O[qkv_offset + (tile_size * i) + (tid * d) + x] *= 1 / row_l_prev;
        }
        L[qkv_offset + (tile_size * i) + (tid * d)] = row_m_new + __logf(row_l_new);
    }
}

__global__ void forward_kernel_optimized(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, int N, int d, int Tc, int Tr, int Bc, int Br, float scale, float* O) {
    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    const int qkv_offset = (bid_x * gridDim.y * N * d) + (bid_y * N * d);
    
    extern __shared__ float sram[];
    const int tile_size = Bc * d;
    
    float* __restrict__ Qi = sram;
    float* __restrict__ Kj = &sram[tile_size];
    float* __restrict__ Vj = &sram[2 * tile_size];
    float* __restrict__ S = &sram[3 * tile_size];

    constexpr int CHUNK_SIZE = 32;  

    for (int i = 0; i < Tr; i++) {
        #pragma unroll
        for (int x = 0; x < d; x += CHUNK_SIZE) {
            #pragma unroll 1
            for (int k = 0; k < CHUNK_SIZE && (x + k) < d; k++) {
                Qi[(tid * d) + x + k] = __ldg(&Q[qkv_offset + (tile_size * i) + (tid * d) + x + k]);
            }
        }
        __syncthreads();

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;

        for (int j = 0; j < Tc; j++) {
            #pragma unroll
            for (int x = 0; x < d; x += CHUNK_SIZE) {
                #pragma unroll 1
                for (int k = 0; k < CHUNK_SIZE && (x + k) < d; k++) {
                    Kj[(tid * d) + x + k] = __ldg(&K[qkv_offset + (tile_size * j) + (tid * d) + x + k]);
                    Vj[(tid * d) + x + k] = __ldg(&V[qkv_offset + (tile_size * j) + (tid * d) + x + k]);
                }
            }
            __syncthreads();

            float row_m = -INFINITY;
            
            #pragma unroll
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                #pragma unroll
                for (int x = 0; x < d; x += CHUNK_SIZE) {
                    float partial_sum = 0.0f;
                    #pragma unroll 1
                    for (int k = 0; k < CHUNK_SIZE && (x + k) < d; k++) {
                        partial_sum = fmaf(Qi[(tid * d) + x + k], Kj[(y * d) + x + k], partial_sum);
                    }
                    sum += partial_sum;
                }
                sum *= scale;
                S[(Bc * tid) + y] = sum;
                row_m = fmaxf(row_m, sum);
            }
            
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l = 0.0f;

            #pragma unroll 1
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tid) + y] = __expf(S[(Bc * tid) + y] - row_m_new);
                row_l += S[(Bc * tid) + y];
            }

            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

            #pragma unroll
            for (int x = 0; x < d; x += CHUNK_SIZE) {
                #pragma unroll 1
                for (int k = 0; k < CHUNK_SIZE && (x + k) < d; k++) {
                    float pv = 0.0f;
                    #pragma unroll
                    for (int y = 0; y < Bc; y++) {
                        pv = fmaf(S[(Bc * tid) + y], Vj[(y * d) + x + k], pv);
                    }
                    const int out_idx = qkv_offset + (tile_size * i) + (tid * d) + x + k;
                    float prev_val = __ldg(&O[out_idx]);
                    O[out_idx] = fmaf(__expf(row_m_prev - row_m_new), prev_val, pv);
                }
            }
            
            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
            __syncthreads();
        }

        const float inv_row_l = 1.0f / row_l_prev;
        #pragma unroll
        for (int x = 0; x < d; x += CHUNK_SIZE) {
            #pragma unroll 1
            for (int k = 0; k < CHUNK_SIZE && (x + k) < d; k++) {
                const int out_idx = qkv_offset + (tile_size * i) + (tid * d) + x + k;
                O[out_idx] *= inv_row_l;
            }
        }
        __syncthreads();
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
    auto L = torch::zeros_like(Q);
    auto m = torch::full({B,nh,N},-INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (4*Bc*d + Bc*Br) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B,nh);
    dim3 block_dim(Bc);

    forward_kernel_optimized<<<grid_dim,block_dim,sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc, Tr, Bc, Br, scale, O.data_ptr<float>());

return O;

}