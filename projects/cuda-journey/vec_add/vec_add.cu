#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdint>

#define CHECK(call)                                                                                                        \
    do {                                                                                                                   \
        cudaError_t err = call;                                                                                            \
        if (err != cudaSuccess) {                                                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                                                                  \
        }                                                                                                                  \
    } while (0)

class GpuTimer {
public:
    GpuTimer() { CHECK(cudaEventCreate(&start_)); CHECK(cudaEventCreate(&stop_)); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start_record() { CHECK(cudaEventRecord(start_)); }
    void stop_record() { CHECK(cudaEventRecord(stop_)); CHECK(cudaEventSynchronize(stop_)); }
    float elapsed_ms() {
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_{}, stop_{};
};

static void init_vec(std::vector<float>& v, float base) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = base + static_cast<float>(i % 1024) * 0.001f;
}

static bool verify_add(const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<float>& out,
                       float atol = 1e-5f) {
    if (a.size() != b.size() || a.size() != out.size()) return false;
    for (size_t i = 0; i < out.size(); ++i) {
        float expected = a[i] + b[i];
        if (std::abs(out[i] - expected) > atol) return false;
    }
    return true;
}

static void vec_add_cpu(const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::vector<float>& out) {
    if (a.size() != b.size() || out.size() != a.size()) return;
    for (size_t i = 0; i < out.size(); ++i) out[i] = a[i] + b[i];
}

// ---Version 1: N need to be less than blockDim.x * gridDim.x
__global__ void vec_add_naive(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// --- Version 2: Grid-stride loop
__global__ void vec_add_grid_stride(const float* a, const float* b, float* c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

// --- Version 3: Vectorized (float4) + grid-stride ---
__global__ void vec_add_float4(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = gridDim.x * blockDim.x;

    int n4 = n >> 2;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    for (int i4 = tid; i4 < n4; i4 += stride4) {
        float4 va = a4[i4];
        float4 vb = b4[i4];
        c4[i4] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }

    int base = (n4 << 2);
    for (int i = base + tid; i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Pick a large size to be bandwidth-bound
    const int N = 1 << 26; // ~= 256 MB per vector
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);
    init_vec(h_a, 1.0f);
    init_vec(h_b, 2.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    const int grid_cap = 65535;
    const int grid_run = (grid > grid_cap) ? grid_cap : grid;

    auto report = [&](const char* name, float ms) {
        // Effective bandwidth: read a,b and write c => 3 * bytes transferred
        double gb = (3.0 * static_cast<double>(bytes)) / 1e9;
        double bw = gb / (ms / 1e3); // GB/s
        std::cout << std::left << std::setw(24) << name
                  << std::right << std::fixed << std::setprecision(3)
                  << ms << " ms, " << std::setprecision(2) << bw << " GB/s" << std::endl;
    };

    std::cout << "Vector size: " << N << " floats (" << std::fixed << std::setprecision(2)
              << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB per vector)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Version 0: CPU ---
    {
        std::vector<float> h_c_cpu(N, 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        vec_add_cpu(h_a, h_b, h_c_cpu);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        report("0. CPU", ms);
        std::cout << "   [Verify: " << (verify_add(h_a, h_b, h_c_cpu) ? "OK" : "FAIL") << "]" << std::endl;
    }

    // --- Version 1: Naive ---
    {
        CHECK(cudaMemset(d_c, 0, bytes));
        timer.start_record();
        vec_add_naive<<<grid_run, block>>>(d_a, d_b, d_c, N);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        report("1. Naive", ms);

        CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        std::cout << "   [Verify: " << (verify_add(h_a, h_b, h_c) ? "OK" : "FAIL") << "]" << std::endl;
    }

    // --- Version 2: Grid-stride ---
    {
        CHECK(cudaMemset(d_c, 0, bytes));
        timer.start_record();
        vec_add_grid_stride<<<grid_run, block>>>(d_a, d_b, d_c, N);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        report("2. Grid-stride", ms);

        CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        std::cout << "   [Verify: " << (verify_add(h_a, h_b, h_c) ? "OK" : "FAIL") << "]" << std::endl;
    }

    // --- Version 3: float4 vectorized ---
    {
        CHECK(cudaMemset(d_c, 0, bytes));
        timer.start_record();
        vec_add_float4<<<grid_run, block>>>(d_a, d_b, d_c, N);
        CHECK(cudaGetLastError());
        timer.stop_record();
        float ms = timer.elapsed_ms();
        report("3. float4", ms);

        CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        std::cout << "   [Verify: " << (verify_add(h_a, h_b, h_c) ? "OK" : "FAIL") << "]" << std::endl;
    }

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}

