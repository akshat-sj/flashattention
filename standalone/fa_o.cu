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

    float* L;
    float* o;
    
    CUDA_CHECK(cudaMalloc(&o,B*nh*N*d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&L,B*nh*N*d * sizeof(float)));

    forward_kernel_optimized<<<grid_dim, block_dim, sram_size>>>(Q, K, V, N, d, Tc, Tr, Bc, Br, scale,o);
    const int output_size = B * nh * N * d;
    float* o_host = new float[output_size];

    CUDA_CHECK(cudaMemcpy(o_host, o, B * nh * N * d * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::endl;
    cudaDeviceSynchronize();
    cudaFree(L);
    cudaFree(o);
}
int main() {
    const int batch_size = 8;
    const int n_head = 16;
    const int seq_len = 1024;
    const int head_embd = 32;

    const int qkv_size = batch_size * n_head * seq_len * head_embd;
    const int o_size = batch_size * n_head * seq_len * head_embd;

    float* Q_host = new float[qkv_size];
    float* K_host = new float[qkv_size];
    float* V_host = new float[qkv_size];
    float* O_host = new float[o_size];

    for (int i = 0; i < qkv_size; ++i) {
        Q_host[i] = static_cast<float>(i % 100); 
        K_host[i] = static_cast<float>(i % 100);
        V_host[i] = static_cast<float>(i % 100);
    }

    float *Q, *K, *V, *O;
    cudaMalloc(&Q, qkv_size * sizeof(float));
    cudaMalloc(&K, qkv_size * sizeof(float));
    cudaMalloc(&V, qkv_size * sizeof(float));
    cudaMalloc(&O, o_size * sizeof(float));

    cudaMemcpy(Q, Q_host, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K, K_host, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V, V_host, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(O, 0, o_size * sizeof(float)); 

    cudaDeviceSynchronize();
    forward(Q, K, V, O, batch_size, n_head, seq_len, head_embd);

    delete[] Q_host;
    delete[] K_host;
    delete[] V_host;
    delete[] O_host;

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);

    return 0;
}