#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define TILE_SIZE 16 // 常用块大小，16x16 或 32x32

#define CHECK(call)                                                                                                       \
    do {                                                                                                                  \
        cudaError_t err = call;                                                                                           \
        if (err != cudaSuccess) {                                                                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                                      \
        }                                                                                                                 \
    } while(0)

void init_matrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = 1.0f;
    }
}

bool verify(const std::vector<float>& C, int M, int N, int K) {
    float expected = (float)K;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C[i] - expected) > 1e-2) return false;
    }
    return true;
}

class GpuTimer {
public:
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void start_record() { CHECK(cudaEventRecord(start)); }
    void stop_record()  { CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop)); }
    float elapsed_ms() {
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
private:
    cudaEvent_t start, stop;
};

#define LAUNCH_KERNEL(kernel, ...) \
    dim3 block(TILE_SIZE, TILE_SIZE); \
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); \
    kernel<<<grid, block>>>(__VA_ARGS__);


__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Shared-memory GEMM with "fused transpose" of B tile:
// Load B[k, col] coalesced from global, but store into shared as Bs[col_local][k_local]
__global__ void gemm_shared_mem(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * TILE_SIZE;
        const int k_a = k_base + threadIdx.x; // k for A load
        const int k_b = k_base + threadIdx.y; // k for B load

        if (row < M && k_a < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k_a];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // transposed
        if (k_b < K && col < N)
            Bs[threadIdx.x][threadIdx.y] = B[k_b * N + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            // Bs is stored as [col_local][k_local], so each thread reads contiguous k.
            sum += As[threadIdx.y][k] * Bs[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void gemm_final_optimized(const float* A, const float* B, float* C, int M, int N, int K) {
    // This version increases arithmetic intensity by having each thread compute 4 C elements.
    constexpr int BM = 16;   // block tile height (M)
    constexpr int BN = 64;   // block tile width  (N)  = TILE_SIZE * 4
    constexpr int BK = 8;    // K tile

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * BM + ty;
    const int j0 = tx * 4;                 // local col within BN
    const int col0 = blockIdx.x * BN + j0; // global col

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * BK;

        if (tx < BK) {
            const int k = k_base + tx;
            As[ty][tx] = (row < M && k < K) ? A[row * K + k] : 0.0f;
        }

        const int linear_tid = ty * blockDim.x + tx;
        const int threads_per_block = blockDim.x * blockDim.y;
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int idx = linear_tid + l * threads_per_block; 
            const int kk = idx / BN; // 0..BK-1
            const int j  = idx - kk * BN; // 0..BN-1
            const int k = k_base + kk;
            const int col = blockIdx.x * BN + j;
            Bs[kk][j] = (k < K && col < N) ? B[k * N + col] : 0.0f;
        }

        __syncthreads();

        // --- Compute ---
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float a = As[ty][kk];
            acc0 = fmaf(a, Bs[kk][j0 + 0], acc0);
            acc1 = fmaf(a, Bs[kk][j0 + 1], acc1);
            acc2 = fmaf(a, Bs[kk][j0 + 2], acc2);
            acc3 = fmaf(a, Bs[kk][j0 + 3], acc3);
        }

        __syncthreads();
    }

    if (row < M) {
        const int base = row * N + col0;
        if (col0 + 0 < N) C[base + 0] = acc0;
        if (col0 + 1 < N) C[base + 1] = acc1;
        if (col0 + 2 < N) C[base + 2] = acc2;
        if (col0 + 3 < N) C[base + 3] = acc3;
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096; // 稍微大一点以便看出差距
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size_A));
    CHECK(cudaMalloc(&d_B, size_B));
    CHECK(cudaMalloc(&d_C, size_C));

    CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);

    GpuTimer timer;
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Matrix Size: " << M << "x" << N << "x" << K << " (Tile=" << TILE_SIZE << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Version 1: Naive ---
    {
        CHECK(cudaMemset(d_C, 0, size_C));
        timer.start_record();
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        std::cout << "1. Naive Global:      " << std::fixed << std::setprecision(2) << ms << " ms, " << gflops << " GFLOPS" << std::endl;
    }

    // --- Version 2: Shared Memory (fused transpose load) ---
    {
        CHECK(cudaMemset(d_C, 0, size_C));
        timer.start_record();
        gemm_shared_mem<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        std::cout << "2. Shared Mem (fused): " << std::fixed << std::setprecision(2) << ms << " ms, " << gflops << " GFLOPS" << std::endl;

        std::vector<float> h_res(M * N);
        CHECK(cudaMemcpy(h_res.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        if (verify(h_res, M, N, K)) std::cout << "   [Result Verified: OK]" << std::endl;
        else std::cout << "   [Result Verified: FAIL]" << std::endl;
    }

    // --- Version 3: Optimized (Shared Mem + Register Blocking) ---
    {
        CHECK(cudaMemset(d_C, 0, size_C));
        timer.start_record();
        // This kernel uses a 16x64 output tile per block (BM=16, BN=64)
        dim3 block4(16, 16);
        dim3 grid4((N + 64 - 1) / 64, (M + 16 - 1) / 16);
        gemm_final_optimized<<<grid4, block4>>>(d_A, d_B, d_C, M, N, K);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        std::cout << "3. Optimized Tiled:   " << std::fixed << std::setprecision(2) << ms << " ms, " << gflops << " GFLOPS" << std::endl;
        
        // 验证结果
        std::vector<float> h_res(M * N);
        CHECK(cudaMemcpy(h_res.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        if (verify(h_res, M, N, K)) std::cout << "   [Result Verified: OK]" << std::endl;
        else std::cout << "   [Result Verified: FAIL]" << std::endl;
    }

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return 0;
}
