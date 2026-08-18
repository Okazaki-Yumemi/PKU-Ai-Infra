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