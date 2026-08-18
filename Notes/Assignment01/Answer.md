# Prob 0.1

代码

```cpp
// 问题 0.1：第一个 CUDA 程序（模块 8 的编译实验也用它）。
// 编译运行：make run/m0_env/01_hello
#include "common.h"

__global__ void hello() {
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<4, 8>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
```

输出

```bash
hello from block 2, thread 0
hello from block 2, thread 1
hello from block 2, thread 2
hello from block 2, thread 3
hello from block 2, thread 4
hello from block 2, thread 5
hello from block 2, thread 6
hello from block 2, thread 7
hello from block 3, thread 0
hello from block 3, thread 1
hello from block 3, thread 2
hello from block 3, thread 3
hello from block 3, thread 4
hello from block 3, thread 5
hello from block 3, thread 6
hello from block 3, thread 7
hello from block 0, thread 0
hello from block 0, thread 1
hello from block 0, thread 2
hello from block 0, thread 3
hello from block 0, thread 4
hello from block 0, thread 5
hello from block 0, thread 6
hello from block 0, thread 7
hello from block 1, thread 0
hello from block 1, thread 1
hello from block 1, thread 2
hello from block 1, thread 3
hello from block 1, thread 4
hello from block 1, thread 5
hello from block 1, thread 6
hello from block 1, thread 7
```

可以见得，CUDA程序的输出是异步执行的，block内的线程在这里是顺序执行的 (因为 8 < warpsize 32,他们会落在一个thread里面)，但block之间的执行顺序是不确定的。

但我跑了四五次都是2 3 0 1 的顺序,有可能是...GPU问题吧


# Prob 0.2

`02_device_query.cu` 的五个空都是 `cudaDeviceProp` 的字段名，对照 CUDA Runtime API 文档补全，然后编译运行。

```cpp
#include "common.h"

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU 型号            : %s\n", prop.name);
    printf("compute capability  : %d.%d\n", prop.major, prop.minor);

    // ====== 空 1：SM 数量（提示：字段名以 multiProcessor 开头） ======
    printf("SM 数量             : %d\n", prop.multiProcessorCount);

    // ====== 空 2：warp 大小 ======
    printf("warp 大小           : %d\n", prop.warpSize);

    // ====== 空 3：每个 block 可用的 shared memory 上限（字节） ======
    printf("shared mem / block  : %zu\n", (size_t) prop.sharedMemPerBlock);

    // ====== 空 4：每个 SM 的最大常驻线程数 ======
    printf("max threads / SM    : %d\n",  prop.maxThreadsPerMultiProcessor);

    // ====== 空 5：全局显存总量（字节） ======
    printf("global mem          : %zu\n", (size_t) prop.totalGlobalMem);

    printf("max threads / block : %d\n", prop.maxThreadsPerBlock);
    return 0;
}
```

Result:

```bash
GPU 型号            : NVIDIA GeForce RTX 5070 Laptop GPU
compute capability  : 12.0
SM 数量             : 36
warp 大小           : 32
shared mem / block  : 49152  # 48 KB
max threads / SM    : 1536
global mem          : 8546484224  # 8150 MB
max threads / block : 1024

```

