#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>
// kernel

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess){ \
        std::fprintf(stderr,"[CUDA ERROR] %s:%d | %s\n", \
        __FILE__,__LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void kernel(const float *x, float *y, int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = y[idx] + 2.0f * x[idx];
    }
}

int main(int argc , char **argv){

    if (argc != 2){
        return 1;
    }

    const int n = std::stoi(argv[1]);

    if (n == 0) {
        std::printf("SUM=%.0f\n", 0.0);
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    CUDA_CHECK(cudaFree(0));


    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);

    for(int i = 0 ; i < n ; i++){
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // ====== 空 4：把 h_a、h_b 拷到 device（注意最后一个方向参数） ======
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1 ) / threads ;
    
    // 计时
    cudaEvent_t start;
    cudaEvent_t stop;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start, 0);
    kernel<<<blocks,threads>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));


    // CPU 计算结果
    double sum = 0.0;
    for (int i = 0 ; i < n ; i++){
        sum += (double)h_y[i];
    }

    printf("SUM=%.0f\n , kernel_ms=%.4f\n", sum,kernel_ms) ;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}