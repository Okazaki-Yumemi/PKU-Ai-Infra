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

# Prob 1.1 (Concept)

(a) 一块标称 100 TFLOPS 的 GPU，执行单条指令的延迟一定低于5GHz的CPU。

> False, GPU的吞吐量由大量核心和高并行度给出，100 TFLOPS 描述的是大量并行运算下的总吞吐能力，不能推出单条指令 latency 更低。

(b) HBM 的“高带宽”指大块连续访问时的吞吐，零散的随机访问达不到标称值。

> True，HBM 的峰值带宽依赖大量、规则且通常较连续的内存访问；零散随机访问很难产生足够高效的 memory transactions 来打满带宽。

(c) 严格串行的迭代算法（每步依赖上一步的结果），即使换一块算力更强的GPU也快不了多少。

> True，GPU的算力主要在于并行的任务，串行的任务难以发挥GPU的真实力量

(d) “算力 1000 TFLOPS”意味着每次运算的延迟是 10−15 秒。

> False,意思是每秒进行1000T次浮点运算。

# Prob 1.2 (Concept)

Session 1 讲座里提过“N 方过百万”这个例子。总计算量$10^12$ FLOP在当代GPU上的运算时间大概是毫秒级，那为什么一个严格在线的串行算法仍然做不到几秒内跑完？（从“延迟”和“吞吐”的角度考虑）

> GPU 的高吞吐主要依赖大量独立任务并行执行，从而用其他 warp 的工作隐藏单次操作的 latency。对于严格在线的串行算法，每一步依赖前一步结果，dependency chain 很窄，没有足够的独立工作用于 latency hiding，因此总时间受到逐步 latency 的限制，而不能直接按照 GPU 的峰值 FLOPS 估算。

# Prob 1.3 (Concept)

| 执行层次        | 软件含义                              | 对应硬件                                  | 直接可用存储                               | 同步与通信                                    |
| ----------- | --------------------------------- | ------------------------------------- | ------------------------------------ | ---------------------------------------- |
| thread      | kernel 的最小 logical execution unit | 一个 lane                               | private registers                    | 自身天然有序                                   |
| warp        | 一组通常 32 个 threads，共同被调度/发射        | SM 内 warp scheduler + execution lanes | 各 thread registers                   | warp-level primitives，如 shuffle          |
| block / CTA | 一组能协作的 threads / warps            | 驻留在一个 SM 上                            | shared memory + per-thread registers | `__syncthreads()` + shared memory        |
| grid        | 一次 kernel launch 的所有 blocks       | 整个 GPU / 多个 SM                        | global memory                        | 普通模型下 block 独立；全局阶段同步通常靠 kernel boundary |

# Prob 1.4 (Concept)

SIMD 与 SIMT 的区别？另：判断正误——Nvidia GPU 在 Volta 之后每个线程有独立的program counter，所以 branch divergence 不再有性能代价。

> SIMD single instruction multi Data, SIMT: single instruction multi Thread : 前者指一个指令，对多个数据来源进行操作，例如向量化计算。 后者是创建多个thread，进行处理.

> SIMD 暴露的是 vector/lane 抽象，一条指令显式作用于多个数据元素；SIMT 暴露的是独立 thread 抽象，程序员写每个 thread 的标量程序，而 GPU 在硬件上把多个 thread 组成 warp，以类似 SIMD 的方式共同发射执行。

> 不行。一个warp内出现分支的时候，所有thread仍然会执行分支内的所有命令，只是mask掉其不属于的分支，这导致计算的效率大大下降了。

# Prob 1.5 (Experiment)

```bash
CPU 单线程      :      9.776 ms  (  2.33 ns/元素)
GPU <<<1, 1>>>  :    249.215 ms  ( 59.42 ns/元素)
GPU <<<1, 256>>>:      6.083 ms  (  1.45 ns/元素)
GPU 铺满 grid   :      0.195 ms  (  0.05 ns/元素, 16384 blocks x 256 threads)
PASS
```

(a) GPU单线程为什么比CPU慢这么多？
> GPU本身的单核计算能力就弱于CPU，但是一个GPU拥有多个核心，让GPU能够同时进行大量的计算，当GPU被限制为单核的时候，其运算速度会大幅度下降

(b)从单block到铺满grid 的提速，说明 GPU 加速计算靠的是什么？
> 单block到铺满grid的提速说明GPU加速靠的是并行度，多核计算

# Prob 1.6

SIMT Simulator ——写一个 warp 的执行模拟器：32 个 lane、共享控制流、带 mask 的分支执行与汇合。请在kernels/simt_sim.py 内完成（docstring 里面有具体实现要求）。

梦回CS61A。

```py
"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def run(program):
    regs = [i for i in range(32)]
    mask = [True for _ in range(32)]

    def helper(program,regs,mask):
        cycle = 0
        op = regs
        
        for instruction in program:
            if instruction[0] == "add":
                k = instruction[1]
                all_false = True
                for i in range(32):
                    if mask[i]:
                        all_false = False
                        op[i] += k
                if all_false != True:
                    cycle += 1
            elif instruction[0] == "mul":
                k = instruction[1]
                all_false = True
                for i in range(32):
                    if mask[i]:
                        all_false = False
                        op[i] *= k
                if all_false != True:        
                    cycle += 1
            elif instruction[0] == "if_lt":
                t = instruction[1]
                then_prog = instruction[2]
                else_prog = instruction[3]

                mask_then = [op[k] <t and mask[k] for k in range(len(op))]

                mask_else = [op[k]>=t and mask[k] for k in range(len(op))]
                
                ans1 = helper(then_prog,op,mask_then)
                ans2 = helper(else_prog,op,mask_else)
                
                cycle += ans1[1]
                cycle += ans2[1]
                

        return (op,cycle)
            
    return helper(program,regs,mask)
```


# Prob 2.1 Fill-in

请填空：m2_first_kernel/01_vector_add.cu ，具体要求见代码注释

```cpp
// 问题 2.1：向量加法（填空）
// 六个空各考一个概念，填完编译运行，"PASS"即可。
// 填完之前这个文件无法通过编译。
#include "common.h"

// ====== 空 1：kernel 需要什么函数修饰符？ ======
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // ====== 空 2：这个线程负责的全局下标 ======
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ====== 空 3：边界保护——总线程数可能多于元素个数 ======
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000003;  // 故意取一个不是 256 整数倍的数
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ====== 空 4：把 h_a、h_b 拷到 device（注意最后一个方向参数） ======
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    // ====== 空 5：block 数——向上取整，保证覆盖全部 n 个元素 ======
    int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;

    // ====== 空 6：启动 kernel（执行配置写在哪里？） ======
    vectorAdd <<<blocksPerGrid,threadsPerBlock>>> (d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
```

# Prob2.2 (Concept)

常用的修饰符:

| 修饰符            | 修饰什么  | 在哪里执行/存放               | 谁能用                    |
| -------------- | ----- | ---------------------- | ---------------------- |
| `__global__`   | 函数    | GPU 上执行                | 通常由 host 用 `<<<>>>` 启动 |
| `__device__`   | 函数/变量 | device                 | device code            |
| `__host__`     | 函数    | CPU 上执行                | host code              |
| `__shared__`   | 变量    | SM 上的 shared memory    | 同一个 block 的 threads    |
| `__constant__` | 变量    | device constant memory | device threads，只读      |
| `__managed__`  | 变量    | Unified Memory         | host/device 都可访问       |




为下列五个场景选择正确的修饰符（如__global__等）。
(a) 在 GPU 上执行、由CPU侧启动的kernel 函数。
> `__global__`, CPU启动在GPU上执行
(b) 只会被 kernel 调用的辅助函数。
> `__device__` 由kernel拿着执行的辅助函数
(c) host 和 device 代码都要调用的小工具函数。
> `__host__ __device__` nvcc 为它生成两个版本,host 和 device 都需要访问的函数
(d) 整个 kernel 运行期间不变、所有线程都要读的系数表。
> `__constant__` SM上所有线程共同读取的小型只读参数 上面共享的不变的东西
(e) block 内线程共享的暂存数组
> `__shared__` SM上共享的东西

# Prob 2.3 (Modify)

`02_vector_add_um.cu` 代码完整，但目前是“显式内存管理”的版本。改之前先按原样编译运行一次，记下耗时——这一版会被你的改动覆盖掉，下面(b)要拿它做对照。然后按文件头的说明改成unified memory 版（cudaMallocManaged），并保持文件头写明的计时窗口不变

然后请回答如下问题：(a)kernel 启动之后、CPU读结果之前，为什么必须有一次同步？在原先的版本里这次同步发生在哪个调用里？(b)对比两版“搬运+kernel+读回”的耗时，分析差距的原因（谁快谁慢都有可能，与使用的卡有关）

before modified:

```bash
搬运 + kernel + 读回: 58.8 ms
PASS
```

改了之后

```bash
搬运 + kernel + 读回: 14.0 ms
PASS
```

具体改动方法:

```cpp
// CPU初始化阶段
    float *h_a ;
    float *h_b ;
    float *h_c ;

    cudaMallocManaged((void**)&h_a,bytes);
    cudaMallocManaged((void**)&h_b,bytes);
    cudaMallocManaged((void**)&h_c,bytes);

    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);


// 不需要CUDA Malloc 和 CUDA MemCopy, 两边用同一个指针

    vectorAdd<<<blocks, threads>>>(h_a, h_b, h_c, n);

    // kernel直接消费h_a h_b h_c 指针
```

> (a) CUDA kernel launch 对 host 是异步的，launch 返回不代表 GPU 已经完成对结果的写入。因此 CPU 在读取 managed memory 中的结果前必须等待 kernel 完成，否则 CPU 可能与 GPU 对同一数据产生未正确排序的访问。在本代码中，这个同步由 CUDA_CHECK_KERNEL() 内部的 cudaDeviceSynchronize() 完成；原显式内存版本随后还有一次同步式 D2H cudaMemcpy。

> (b) 在本机 RTX 5070 Laptop 的测试中，显式内存管理耗时 58.8 ms，而 Unified Memory 为 14.0 ms，约快 4.2×。Unified Memory 并没有消除 CPU 与 GPU 之间的数据移动，而是把显式 cudaMemcpy 改成由 CUDA runtime/driver 管理的页面迁移和访问。当前平台上这种 managed-memory 路径的开销明显低于原版使用普通 pageable host allocation 加显式 memcpy 的路径，因此测得更快。但该结果依赖 GPU、驱动和系统环境，不能认为 Unified Memory 普遍比显式内存管理更快。


# Prob 2.4
判断对错，可以顺带补一句理由

(a) vectorAdd<<<...>>>(...) 这条语句返回时，kernel 一定已经执行完毕。
> 错。返回给 CPU 时，不保证任何一个 GPU thread 已经执行完，甚至不能据此判断 kernel 已经开始执行到什么程度。 Kernel launch 对 host 通常是异步的，launch 语句返回只表示 kernel 已经被提交，不代表 kernel 已经执行完毕；如果 host 后续必须等待 GPU 结果，需要显式同步，例如 cudaDeviceSynchronize()

(b) 同一个 stream 里，cudaMemcpy（device 到 host）会等它前面的 kernel 全部完成后才开始拷贝。
> 会，同一个 stream 里，cudaMemcpy（device → host）会等它前面的 kernel 全部完成后才开始拷贝, 原因是同一个 stream里面的操作具有顺序关系,后面的操作不能因为： “现在 HBM bandwidth 好像有空”

(c) kernel 内部的非法访存，会在启动语句处同步地报出来。
> 不会，得进入到内核实际执行的时候发现才会报错。然后错误会被传回.

# Prob 2.5 Debug
修bug：03_bug_launch.cu，详细内容见相关文件

原来的代码:

```cpp
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));


    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    int threads = 2048;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();
    // 注意：这里故意没有做任何错误检查。

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
```

执行:

```bash
make run/m2_first_kernel/03_bug_launch
nvcc -O2 -std=c++17 -I. -arch=native -o bin/m2_first_kernel/03_bug_launch m2_first_kernel/03_bug_launch.cu
./bin/m2_first_kernel/03_bug_launch
CUDA error cudaErrorInvalidConfiguration at m2_first_kernel/03_bug_launch.cu:36: invalid configuration argument
make: *** [ Makefile:24: run/m2_first_kernel/03_bug_launch ] Error 1
rm bin/m2_first_kernel/03_bug_launch
```

把thread数目修改到256后恢复正常。

提示说: **// 程序一声不吭？（问题 0.2 打印过的哪个上限和这里有关？）**

回看发现: max threads / block : 1024，改到1024 pass 
调整到1025之后则失败.

一开始没加 CUDA_CHECK_KERNEL() 的时候，kernel会直接不执行如果不去查error，就会往下走.因为前面由cudaMemset(d_c,0,bytes) 使得c里面全0,没有报错而是assertion error

```cpp
#define CUDA_CHECK_KERNEL()             \
    do {                                \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
```

这个地方显示的就是这样，使得cudaError能够被捕获

# Prob 2.6

04_matrix_add.cu 用二维的 block 和 grid 处理 1000×700 的矩阵，请填空实现矩阵加法。

```cpp
// 问题 2.6：二维矩阵加法（填空）。
// 用二维的 block 和 grid 处理 M x N 矩阵，四个空都和二维索引有关。
// 填完之前这个文件无法通过编译。
#include "common.h"

__global__ void matrixAdd(const float *a, const float *b, float *c, int M, int N) {
    // ====== 空 1：这个线程负责的行号（用 y 方向的内建变量） ======
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // ====== 空 2：这个线程负责的列号（用 x 方向的内建变量） ======
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // ====== 空 3：二维边界保护 ======
    if (row < M && col < N) {
        int idx = row * N + col;  // 行优先展开成一维下标
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int M = 1000, N = 700;  // 都不是 16 的整数倍
    const long total = (long)M * N;
    size_t bytes = total * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, total, 1);
    fill_random(h_b, total, 2);
    for (long i = 0; i < total; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);  // x 方向 16 列，y 方向 16 行
    // ====== 空 4：二维 grid——两个方向都要向上取整 ======
    int block_row = (N + 16 - 1) / 16;
    int block_col = (M + 16 - 1) / 16;
    dim3 blocks(block_row, block_col);
    matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, total));
    return 0;
}
```

一开始写错了，给行写成M了，实际上 M x N 的矩阵，行大小是 N，改了之后就pass了


# Prob2.7 Modify

05_grid_stride.cu 的 launch 被固定成 <<<64, 256>>>，线程总数远小于n，当前FAIL。在launch
配置不变的前提下，把kernel 改成grid-stride loop，让任意 n 都能 PASS。

```cpp
// 问题 2.7：grid-stride loop（改造题）。
// 现状：launch 只给了 64 个 block，线程总数远小于 n，所以输出 FAIL。
// 任务：不许改 launch 配置，把 kernel 改成 grid-stride loop——每个线程
//      跨过整个 grid 的步长处理多个元素——让任意 n 都能 PASS。
// 参考：NVIDIA 博客 "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
//      https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x ; i < n ; i += blockDim.x * gridDim.x){
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1<<24;  // 16M 元素，远多于 64 * 256 = 16384 个线程
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    vectorAdd<<<64, 256>>>(d_a, d_b, d_c, n);  // launch 配置不许动
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
```

然后请回答——这种写法的价值在哪里？launch 只有 16384 个线程时，性能上要付出什么代价

> 价值就是flexible，可以让一套kernel适应不同的硬件，问题是这样相当于把一个大任务切开，有了部分串行的性质

## Prob 2.8
运行下面的程序两三次，观察16个block打印输出的先后。

代码:
```cpp
#include "common.h"

__global__ void whoami() {
    // 让每个 block 的 0 号线程报到。
    if (threadIdx.x == 0) {
        printf("block %d 报到\n", blockIdx.x);
    }
}

int main() {
    whoami<<<16, 32>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
```

执行:
```bash
block 14 报到
block 12 报到
block 15 报到
block 13 报到
block 10 报到
block 8 报到
block 11 报到
block 6 报到
block 9 报到
block 7 报到
block 4 报到
block 2 报到
block 5 报到
block 0 报到
block 3 报到
block 1 报到

=====================

block 14 报到
block 12 报到
block 15 报到
block 13 报到
block 10 报到
block 8 报到
block 11 报到
block 6 报到
block 9 报到
block 7 报到
block 4 报到
block 2 报到
block 5 报到
block 0 报到
block 3 报到
block 1 报到

==================

block 14 报到
block 12 报到
block 15 报到
block 13 报到
block 10 报到
block 8 报到
block 11 报到
block 6 报到
block 9 报到
block 7 报到
block 4 报到
block 2 报到
block 5 报到
block 0 报到
block 3 报到
block 1 报到
==================
```

(a) 顺序由谁决定？
> 不同 block 何时被调度到哪个 SM、以什么先后顺序执行，由 GPU 的硬件/runtime 调度机制决定，程序员不能指定或依赖这个顺序。

(b)程序的正确性可以依赖block的执行顺序吗？这条限制和Guide1.1说的scalable programming model 有什么关系？

> 程序正确性不能依赖不同 block 的执行顺序。CUDA 将 block 设计成可独立调度的工作单元，因此无论 GPU 有多少 SM，都可以按任意顺序、并发程度和批次把 blocks 映射到 SM 上。正是这种 block independence 使同一个 kernel 能自动扩展到不同规模的 GPU，这就是 scalable programming model 的重要基础。


# Prob 2.9 (From Scratch)

在`m2_first_kernel/` 下写出完整的 CUDA 程序 saxpy.cu，实现 y ←2.0·x+y（单精度）。要求如下:

- 不允许include common.h。错误检查宏和 cudaEvent 计时都要自己写一遍。
- 命令行用法：./saxpy <n>，n 是元素个数。输入数据按固定公式生成（都是float）：x[i]= ((i % 2048)- 1024) * 0.5f，y[i] = (i % 1024)- 512。
- kernel 算完把 y 拷回 host，用 double 累加所有 y[i]，输出一行 SUM=< 总和 >（用printf("SUM=%.0f\n", s) 这样的格式，同一行里可以再带上n和kernel毫秒数），SUM结果将用于对拍检验程序正确性，exitcode 应为0
- n=0时输出SUM=0，exit code 为 0（0 个 block 的 kernel launch 是非法的，特判即可）

判测脚本覆盖n∈{0,1,31,1024,1025,2^20,2^20+3}，命令如下
```bash
cd assignment01/cuda/m2_first_kernel
./judge_saxpy.sh saxpy.cu
```


```cpp
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

```

# SIMT 执行

# Prob 3.1

设blockDim = (8, 8, 1)。
(a) threadIdx = (3, 5, 0) 的线性编号是多少？它在第几个 warp、warp 内第几个 lane？
> linear tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z = 3 + 8 x 5 = 43 -> warp_id = 1 , lane_id  = 11
(b) 这个 block 一共占多少个 warp？
> 这个block共有 8 * 8 / 32 = 2 个warp
(c) 若 blockDim = (33, 1, 1)，占几个 warp？这样配置浪费在哪里？
> 占用 2 个warp ， 其中第二个warp几乎全空，浪费很多空间

# Prob 3.2

m3_simt/01_divergence.cu 的两个 kernel 每线程计算量相同，分支划分不同——一个按 thread编号的奇偶分（同一个warp里一半一半），一个按warp边界对齐分。请先预测一下哪个版本运行会更快一点，大概快几倍，然后运行验证：

> 预测: warp边界对齐，只用执行4个， 分支组可以等价认为执行了8个，所以时间大约是2倍。

```bash
warp 内分支 (tid % 2)    :    1.455 ms
按 warp 分支 (tid/32 % 2):    0.749 ms
比值: 1.94

warp 内分支 (tid % 2)    :    1.448 ms
按 warp 分支 (tid/32 % 2):    0.752 ms
比值: 1.93

warp 内分支 (tid % 2)    :    1.459 ms
按 warp 分支 (tid/32 % 2):    0.774 ms
比值: 1.88
```

请解释实测比值，并回答——若两个分支的计算量一大一小，按thread 编号奇偶分的 kernel和按warp 边界对齐分的kernel 的运行时间分别由什么决定

> 实验比率低于理论比率，估计是因为 warp内分支和按照warp分支都会有相对应的 "调度、打包开销"等共同的固定开销。
> 二者运行时间最显著的差异来源于divergence导致的同一warp内等待的问题

> 两个分支计算量一大一小的话，奇数、偶数分开的计时时间 = T_long + T_short ， 合并的是 T_long

# Prob 3.3

02_sync_matters.cu 让每个 block 用 shared memory 把自己的 256 个元素倒序。请按文件开头的注释内容进行实验。

```cpp
// 问题 3.3：__syncthreads 实验。
// 每个 block 把自己的 256 个元素倒序：先搬进 shared memory，同步，
// 再交叉着读出来。任务：
//   1. 直接运行，确认 PASS；
//   2. 注释掉 __syncthreads() 那一行，再运行几次，观察结果；
//   3. 回答 handout 里的问题。

#include "common.h"

#define BLOCK 256

__global__ void reverse_blocks(const float *in, float *out, int n) {
    __shared__ float buf[BLOCK];
    int base = blockIdx.x * BLOCK;
    int t = threadIdx.x;

    buf[t] = in[base + t];
    //__syncthreads();  // <-- 实验对象
    out[base + t] = buf[BLOCK - 1 - t];
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 7);
    for (int b = 0; b < nblocks; b++)
        for (int t = 0; t < BLOCK; t++)
            h_ref[b * BLOCK + t] = h_in[b * BLOCK + (BLOCK - 1 - t)];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reverse_blocks<<<nblocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_out, h_ref, n));
    return 0;
}
```

1. 直接运行

```BASH
PASS
```

2. 注释掉

```BASH
MISMATCH at 0: got 0.000000, want 4.820000
FAIL
```

(a) 为什么注释掉 sync 后代码不能正确地运行？

> 观察可知，所有线程都会读取 shared buffer， 然后先把in里面的数据拷贝到buffer里面，最后再将buffer里面的数据倒过来放到out内。 注释掉sync之后，相当于buffer还没有准备就绪，就从里面读取数据送到out内。我们应该等待buf数组读取了所有的数组再倒过来

(b) (Optional) 注释掉 sync 后，翻转后的数组错的位置比较随机，但是有些位置一直是对的，试解释原因。（tip: 算一算t与255−t有没有可能落在同一个warp
> warp分别是 0~31 32~63 64 ~ 95 96 ~ 127 128 ~ 159 160 ~ 191 192 ~ 223 224 ~ 255
> 不过 t = 127 时候 255-t = 128 ， 二者不可能在同一个warp，所有warp都是反过来的？
> 为什么有些位置一直是对的? 假设 warp 0 <-> warp 7, 可能 warp 0先写好了， warp 7 就能全对

# Prob 3.4

__syncthreads 只能同步本 block 内的 threads，那需要全 grid 同步时，标准做法是什么？

> kernel boundary / 拆成多个kernels, cooperative Groups 的 grid.sync() 可以在满足cooperative launch 等特殊条件时实现kernel 内grid-wide synchronization

# Prob 3.5 （FROM-SCRATCH）：block 内归约


在03_reduce.cu 里从零实现两个求和归约kernel（判测与计时的代码已经写好）。两个kernel的contract 见文件头。PASS 后，试解释实测性能差距的原因.

（Optional）基于两点事实——
(a) __shfl_down_sync 是 warp 内寄存器级别的线程间数据交换指令，自带同步效果且延迟极小；
(b)归约到最后32个元素后，活跃线程若都落在同一个warp里，就不再需要__syncthreads（可以想想为什么）——据此试写出第三版优化后的kernel。测试时只会跑前两版，第三版自己在main里照着加一次run_one调用即可。

```cpp
__global__ void reduce_interleaved(const float *in, float *out) {
    // TODO：从这里开始写（交错配对版本）
    __shared__ float buf[BLOCK];
    int tid = threadIdx.x;

    buf[tid] = in[tid + blockIdx.x*BLOCK];
    
    int s = 1;
    while( s <= blockDim.x /2){
        __syncthreads();
        if(tid % (2*s) == 0){
            buf[tid] += buf[tid + s];
        }
        s *= 2;
        __syncthreads();
    }
    if(tid == 0){
        out[blockIdx.x] = buf[0];
    }
}

__global__ void reduce_contiguous(const float *in, float *out) {
    // TODO：从这里开始写（连续配对版本）
    __shared__ float buf[BLOCK];
    int tid = threadIdx.x;
    buf[tid] = in[tid + blockIdx.x*BLOCK];
    int s = blockDim.x / 2;
    while(s > 0){
        __syncthreads();
        if(tid < s){
            buf[tid] += buf[tid + s];
        }
        s/= 2;
        __syncthreads();
    }
    if(tid == 0){
        out[blockIdx.x] = buf[0];
    }
}

__global__ void shuffle_kernel(const float *in , float *out){
    __shared__ float buf[BLOCK];
    int tid = threadIdx.x;
    buf[tid] = in[tid + blockIdx.x * BLOCK];
    int s = blockDim.x / 2;
    while(s > 32){
        __syncthreads();
        if(tid < s){
            buf[tid] += buf[tid + s];
        }
        s/= 2 ;
        __syncthreads();
    }

    if(tid < 32){
        float v = buf[tid] + buf[tid+32];

        for(int offset = 16; offset > 0; offset/=2 ){
            v += __shfl_down_sync(0xffffffff, v, offset);
        }

        if(tid == 0){
            out[blockIdx.x] = v;
        }
    }
}
```

> 本机跑出来的时间比较奇怪，interleaved / contiguous 在 0.8 ~ 1.3 都有，估计是本机实验架构的问题，放到A100测试得到的时间就是

- contiguous_ms = 0.0169
- interleaved_ms = 0.0343
- shuffle_ms = 0.0101

加速比是 2

测试均通过。完成。


# 4. 存储空间

# Prob 4.1

补全下表:

| 空间            | 谁可见                        | 生命周期                           | 片上 / 片外                    | 谁管理                |
| ------------- | -------------------------- | ------------------------------ | -------------------------- | ------------------ |
| register      | 单个线程                       | 线程                             | **片上**                     | 编译器                |
| local         | 单个线程                       | 线程 / kernel 执行期间               | **片外**                     | 编译器                |
| shared        | 一个 block                   | block                          | **片上**                     | 程序员 / kernel       |
| global        | 整个 grid，且可跨 kernel 使用      | **application / 直到释放**         | **片外**                     | 程序员 / CUDA Runtime |
| constant      | 整个 grid，只读                 | **application / CUDA context** | **片外**，但有片上 constant cache | 程序员 / CUDA Runtime |
| L1 / L2 cache | L1：单个 SM；L2：整个 GPU 各 SM 共享 | 硬件动态，无程序员可依赖的内容生命周期            | **片上**                     | GPU 硬件             |


# Prob 4.2

请填空：m4_memory/01_stencil.cu ，具体要求见代码注释。

```cpp
__global__ void stencil_static(const float *in, float *out, int n) {
    // ====== 空 1：静态 shared 数组，要装下 BLOCK 个元素加两侧 halo ======
    __shared__ float tile[BLOCK+2];

    int g = blockIdx.x * blockDim.x + threadIdx.x;  // 全局下标
    int l = threadIdx.x + RADIUS;                   // 在 tile 里的位置

    tile[l] = (g < n) ? in[g] : 0.f;
    // 块两端的线程多搬一个 halo 元素。 //用 第一个来整体处理，方便
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }

    // ====== 空 2：在这里补上一行 ======
    /* 填这里 */
    __syncthreads();

    if (g < n) {
        // ====== 空 3：用 tile（不许用 in）算三点平均 ======
        out[g] = (tile[l-1] + tile[l] + tile[l+1])/3.f;
    }
}

__global__ void stencil_dynamic(const float *in, float *out, int n) {
    // ====== 空 4：动态 shared 数组的声明方式（大小在 launch 时才给出） ======
    /* 填这里（声明动态 shared 数组 tile）*/

    extern __shared__ float tile[];

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + RADIUS;

    tile[l] = (g < n) ? in[g] : 0.f;
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }
    __syncthreads();
    if (g < n) {
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

    // ====== 空 5：动态 shared 版本的 launch——第三个参数该填多少字节？ ======
    stencil_dynamic<<<blocks, BLOCK, (BLOCK+2) * sizeof(float)>>>(d_in, d_out, n);
```


CUDA里面动态 shared memory 的写法是 extern __shared__ float tile[];

大小由 kernel<<<blocks, BLOCK, ???>>>(...); 里面的??? 给出