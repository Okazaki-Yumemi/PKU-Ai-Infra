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

```text

static shared:

__shared__ float tile[258];
                   ↑
             大小写在 kernel 里


dynamic shared:

extern __shared__ float tile[];
                         ↑
                    kernel 里不写大小

kernel<<<..., ..., 258*sizeof(float)>>>
                    ↑
              launch 时决定大小
```

# Prob 4.3 MODIFY

02_constant_coeff.cu ：按文件头的说明进行修改。修改完后测试

```cpp
// 问题 4.3：把系数表搬进 constant memory（ MODIFY ）。
// 下面的 poly_eval_global 把 8 个多项式系数放在 global memory，每个线程读 8 次。
// 它保留不动，作为对比基准。
// 任务：
//   1. 声明 __constant__ float COEF[8]；
//   2. 在 main 里标了 TODO 的地方用 cudaMemcpyToSymbol 把系数拷进去；
//   3. 把 poly_eval_const 写成读 COEF 的版本——参数表保持不变（判测代码要用
//      同一个函数指针类型跑两版），里面不再用 coef 这个指针即可。
// 两版都要 PASS。评测结果会包含两版的耗时和比值。
```

在文件外面 (文件作用域) 声明 `__constant__ float COEF[8] `

然后用 `CUDA_CHECK(cudaMemcpyToSymbol(COEF, h_coef, sizeof(h_coef)))` 拷贝.


# Prob 4.4 Concept

判断对错，可以顺带补一句理由。

(a) loacl memory 的 "local" 指作用域私有，它实际上在片外显存里
> 对，local memory 的 “local” 指的是每个 thread 私有的地址空间，而不是物理位置；其 backing storage 位于片外 device memory（显存）中，访问也可能经过 L1/L2 cache。
(b) 对数组用运行期才知道的下标做索引，可能迫使它被放进 local memory
> 对。如果数组使用运行期才能确定的下标，编译器可能无法把数组元素映射到独立的 registers，因此可能把该数组放入 local memory。

# Prob 4.5 Fill-in
请填空：03_histogram.cu ，具体要求见代码注释。

```cpp
// 问题 4.5：直方图（填空）。
// 统计 16M 个字节的值落在 256 个 bucket 里的次数。
// 注意：多个线程可能同时修改同一个 bucket 的值。

__global__ void histogram(const unsigned char *data, unsigned int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        unsigned char v = data[i];
        // ====== 空 1：往 hist[v] 里加 1
        //         该用哪个原子操作？ ======
        /* 填这里 */;
        atomicAdd(&hist[v],1);
    }
}
```

atomicAdd 往一个地址上面的值增加等等。

```bash
平均耗时 2.4442 ms  (6.86 GB/s)
PASS

平均耗时 2.4756 ms  (6.78 GB/s)
PASS

平均耗时 2.4486 ms  (6.85 GB/s)
PASS
```


# Prob 4.6 Modify

04_histogram_priv.cu ：请按要求修改代码。改完测试指令

```cpp
__global__ void histogram_naive(const unsigned char *data, unsigned int *hist,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

__global__ void histogram_priv(const unsigned char *data, unsigned int *hist,
                               int n) {
    // TODO：从这里开始写（shared memory 私有化版本）
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int stride = blockDim.x * gridDim.x;
    __shared__ unsigned int cnt[BINS];
    cnt[threadIdx.x] = 0;
    __syncthreads();
    for (; i < n; i += stride) {
        atomicAdd(&cnt[data[i]], 1u);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        for(int k = 0 ; k < BINS ; k++){
            atomicAdd(&hist[k], cnt[k]);
        }
    }
}
```
```bash
naive: PASS  平均 2.4448 ms  (6.86 GB/s)
priv : PASS  平均 0.0783 ms  (214.31 GB/s)
naive / priv = 31.23x
```

一定要两个syncthreads,然后我发现还可以再优化

```cpp
if(threadIdx.x == 0){
        for(int k = 0 ; k < BINS ; k++){
```

不如改成
```cpp
atomicAdd(&hist[threadIdx.x], cnt[threadIdx.x]);
```

```bash
naive: PASS  平均 2.4695 ms  (6.79 GB/s)
priv : PASS  平均 0.0367 ms  (457.20 GB/s)
naive / priv = 67.30x
```

# Prob 4.7 Experiment

运行实验，并根据实验数据填表


```cpp
// stride = 1 时是连续访问；stride 变大后，warp 里相邻线程读的地址
// 相距 stride 个 float。n 是 2 的幂，& (n-1) 等价于取模。
__global__ void strided_copy(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (long)i * stride & (n - 1);
        out[i] = in[j];
    }
}
```
实验结果:

|Stride | 1 | 2 | 4 | 8 | 16 | 32 |
| ----- | - | - | - | - | -  | -  |
| GB/s  |334.0| 219.6 | 131.4 | 77.5 | 78.3 | 78.1|

stride = 1 的时候，对于 i = 0, 1, 2, 3, 4, 5而言 (n = 1 << 24 取模看作是自己)

j = 0, 1, 2, 3 , 4, 5

读取 j 是连续的

对于stride = 16

j = 0 , 16, 32, 48, 64, 80

相隔距离很大, 数据访问相隔很大，warp 内 global memory access 的 coalescing（合并访问）变差。

stride≈8 后，memory transaction utilization 已经接近很差的状态，再继续把地址拉开，也没有多少新的损失空间了。

# Prob 4.8 (EXPERIMENT)

```cpp
// 问题 4.8：occupancy 实验。
// 思路：shared memory 按 block 分配，一个 block 占得越多，SM 上能同时
// 驻留的 block 就越少，常驻 warp 数（occupancy）随之下降。下面的 kernel
// 声明了不实际用于储值的动态 shared memory——计算量和访存量完全不变，
// 变的只有 SM 上的并行度。
// 程序对每一档 shared memory 用量做两件事：
//   1. 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询这一档下
//      每个 SM 理论上能驻留几个 block，换算成 occupancy；
//   2. 实测同一个逐元素加法 kernel 的有效带宽。
// 记录实测数据，并回答相关问题

#include "common.h"

#define BLOCK 256

__global__ void stream_add(const float *a, const float *b, float *c, int n) {
    extern __shared__ float ballast[];  // 只占 shared memory，不使用
    (void)ballast;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int smem_sm = (int)prop.sharedMemPerMultiprocessor;
    int smem_blk_max = (int)prop.sharedMemPerBlockOptin;
    int max_threads = prop.maxThreadsPerMultiProcessor;
    printf("%s：shared memory %d KB / SM，最大常驻 %d 线程 / SM\n\n",
           prop.name, smem_sm / 1024, max_threads);

    // 允许单个 block 申请超过默认上限（48 KB）的动态 shared memory
    CUDA_CHECK(cudaFuncSetAttribute((const void *)stream_add,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_blk_max));

    const int n = 1 << 26;
    size_t bytes = (size_t)n * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));
    int nblocks = (n + BLOCK - 1) / BLOCK;

    // shared memory 档位：按每 SM 总量的比例给定。一个 block 占了总量的
    // 1/x，每 SM 大致就只能驻留 x 个 block。下面六档挑得能在多数卡上落到
    // 六个不同的驻留块数，但实际落点还受架构影响（有的架构给每个 block
    // 额外保留一小块 shared），一切以 API 报出来的数为准。
    const double fracs[] = {0.0, 0.18, 0.23, 0.31, 0.44, 0.55};
    printf("%-14s %-16s %-11s %s\n",
           "shared/block", "理论 block/SM", "occupancy", "实测带宽");
    for (int k = 0; k < 6; k++) {
        int smem = (int)(smem_sm * fracs[k]);
        if (smem > smem_blk_max) smem = smem_blk_max;

        int active = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, stream_add, BLOCK, smem));
        double occ = 100.0 * active * BLOCK / max_threads;

        stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);  // 热身
        CUDA_CHECK_KERNEL();
        const int reps = 20;
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;

        printf("%8.1f KB %10d %14.1f%% %10.1f GB/s\n",
               smem / 1024.0, active, occ, gbps);
    }

    // 讲义里提到的另一个 API：让 runtime 建议一个 occupancy 最高的 block size
    int min_grid = 0, best_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &best_block, stream_add, 0, 0));
    printf("\ncudaOccupancyMaxPotentialBlockSize 建议（smem = 0 时）：blockSize = %d\n",
           best_block);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
```


实验结果
```bash

NVIDIA GeForce RTX 5070 Laptop GPU：shared memory 100 KB / SM，最大常驻 1536 线程 / SM

shared/block   理论 block/SM  occupancy   实测带宽
     0.0 KB          6          100.0%      346.2 GB/s
    18.0 KB          5           83.3%      332.4 GB/s
    23.0 KB          4           66.7%      339.8 GB/s
    31.0 KB          3           50.0%      335.9 GB/s
    44.0 KB          2           33.3%      297.5 GB/s
    55.0 KB          1           16.7%      176.4 GB/s

cudaOccupancyMaxPotentialBlockSize 建议（smem = 0 时）：blockSize = 768
```

(a) 用程序开头打印的“shared memory / SM”和“最大常驻线程/SM”，手算其中一个的驻留block 数和 occupancy，和 API 的结果对照。

> 程序中每个block = 256 threads. 例如选择 31.0 KB / block 这个， 100/31 = 3, 只能放3个block, 3x256 = 768 个thread，只占用了一半

(b) 带宽为什么随 occupancy 下降？用“延迟隐藏需要足够多的常驻warp”组织你的解释。

> occupancy 降低意味着每个 SM 上能够同时 resident 的 warps 变少。当一个 warp 因 global memory access 等待时，warp scheduler 可以切换执行的其他 ready warps 也随之减少，因此隐藏 memory latency 的能力下降。当 resident warps 少到不足以持续填满 memory pipeline 时，stall 被暴露出来，有效带宽下降。

(c)表中带宽随occupancy单调下降，但明显不成正比——从100%到75%带宽掉了多少？从37.5%到12.5%又掉了多少？试解释这个差别。

>带宽并不与 occupancy 成正比。在本机上，从 100% occupancy 降到 66.7%，带宽仅从 346.2 GB/s 降到 339.8 GB/s，约下降 1.8%，说明此时 resident warps 仍然足以隐藏 memory latency，并基本饱和 memory pipeline。相反，从 33.3% 降到 16.7% 时，带宽从 297.5 GB/s 降到 176.4 GB/s，约下降 40.7%。此时 resident warps 太少，当部分 warps 等待 memory 时，scheduler 缺少其他 ready warps 可执行，latency 无法被充分隐藏，因此性能快速下降。因此 occupancy 是 latency-hiding capacity 的指标，而不是性能或带宽的线性比例。



# 5. 计时与异步初步

# Prob 5.1
运行实验： 

并回答下列问题：
(a) 哪个数值可以当作kernel 耗时写进报告？
(b) 另外两个各具体测的是什么？

```bash
host 计时、不等 GPU :     0.0127 ms
host 计时、等 GPU   :     0.2544 ms
cudaEvent 计时      :     0.2632 ms
```

代码:

```cpp

    // 方式一：host 计时，启动后立刻停表。
    auto t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    auto t1 = std::chrono::steady_clock::now();
    double ms_nosync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaDeviceSynchronize());

    // 方式二：host 计时，等 GPU 干完再停表。
    t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::steady_clock::now();
    double ms_sync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 方式三：cudaEvent 计时。
    GpuTimer timer;
    timer.start();
    busy<<<blocks, threads>>>(d_out, iters);
    float ms_event = timer.stop_ms();
```

(a) kernel耗时是cudaEvent计时

(b) 不同步的 host 计时主要测 kernel launch/enqueue 的 host 端开销，并未等待 GPU 完成；同步的 host 计时测从 host 发起 kernel 到 cudaDeviceSynchronize() 确认 GPU 完成的端到端 wall-clock 时间，因此还包含 launch 和同步等 host/runtime 开销。

# prob 5.2 （CONCEPT）
判断对错，可以顺带补一句理由。
(a) 同一个 stream 里的操作按提交顺序执行。
> 对，同一个stream内按照顺序执行，由host提交请求

(b) kernel 启动后，host 代码立刻继续往下执行。
> 对。 CUDA kernel launch 通常相对于 host 是异步的；host 只等待 launch 被提交并返回，不等待 GPU 完成 kernel。

(c) unified memory 下，CPU 访问一页正被 GPU 占用的内存，会触发缺页与页迁移
> 对 Unified Memory 由系统维护 CPU/GPU 间的数据一致性；当 CPU 访问当前驻留在 GPU 一侧的 managed page 时，可能通过 page fault 触发页面迁移和相应的同步/一致性处理。


# 6 Tile 视角

Guide 从 13.x 起把 tile 编程作为与 SIMT 并列的第二种官方模型写进正文。tile 模型描述的是“一个block 对一块数据做什么”，而 block 内 threads 的分工由编译器决定。此 Module 只做概念铺垫和对照阅读。

# TileLang 与 Triton

# Prob 7.1 (Fill-in)

请填空：`kernels/vector_add.py`

```py
"""问题 7.1：Triton 向量加法（填空）。

四个空对应 Triton kernel 的四个 basic operation。填完运行：
    pytest tests/test_vector_add.py
没有 GPU 也能跑，conftest.py 会自动切到 interpreter 模式。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    # ====== 空 1：当前 program 在一维 grid 里的编号 ======
    pid = tl.program_id(0)
    # ====== 空 2：这个 program 负责的一段全局下标（长度 BLOCK_SIZE） ======
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    # ====== 空 3：屏蔽越界位置的 mask ======
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # ====== 空 4：把 x + y 写回 z（别忘了 mask） ======
    z = x +  y
    tl.store(z_ptr + offsets,z ,mask= mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z
```

和之前差不多，pid读取之后作为block id，triton是block层级的操作.

设置好offsets和mask之后就能读取，操作

# prob 7.2 （MODIFY）

按文件开头注释要求修改：kernels/fused_op.py 

```py
"""问题 7.2：fused elementwise（改造题）。

scale_kernel 目前功能完整，相应代码不要变动。
fused_kernel 目前和 scale_kernel 完全一致，是你需要修改的 kernel。
任务：改成 z = relu(a * x + b)，其中 a、b 是标量。
TIP: 只需要动计算那一行，再把 a、b 传进 kernel——主体不变，
这正是 Tile 视角的好处:-)。改完运行：
    pytest tests/test_fused_op.py
"""

@triton.jit
def fused_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr, a, b):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = tl.maximum(0, a*x + b)# TODO：改成 relu(a * x + b)，提示 tl.maximum
    tl.store(z_ptr + offsets, z, mask=mask)


def fused(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE,a = a,b = b)  # TODO：把 a、b 传进去
    return z
```
改完回答——与Module 2 里改 CUDA kernel 相比，这次的改动主要集中在 kernel 的什么部
分？主体代码为什么一行都不用动？

> 改动主要集中在 tile 内的逐元素计算表达式，即把原来的 x * 2 改成了 relu(a * x + b)，同时增加了标量参数 a、b。因为输入输出的 shape 和“一一对应的 elementwise 映射”没有改变，所以 program grid、offsets、mask、load 和 store 的数据访问模式都可以保持不变。Triton 把一个 program 对一整个 tile 的索引和数据搬运与 tile 内具体执行的 elementwise computation 分离开，因此只需要修改计算部分。

# Prob 7.3

请填空：kernels/tilelang_scale_add.py ，具体要求见代码注释

```py
"""问题 7.3：TileLang 版 scale-add（填空）。

Y = 2 * X + 1，X 形状 (M, N)。两个空对应 TileLang 的两个 basic operation。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k scale_add
"""

def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid——x 方向要多少个 block（管 N 列），
        #         y 方向要多少个（管 M 行）？提示：T.ceildiv ======
        x_blocks = T.ceildiv(N,block_N)
        y_blocks = T.ceildiv(M,block_M)
        with T.Kernel(x_blocks, y_blocks, threads=128) as (bx, by):
            # ====== 空 2：block 内并行遍历 tile 的每个元素，
            #         提示：T.Parallel(维度1, 维度2) ======
            for i, j in T.Parallel(block_M,block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add
```

首先拿到x,y block的数目，然后交给内核去管理，利用by\*block_M确定行，bx\*blockN 确定列


# prob 7.4 （FILL-IN）

请填空：kernels/tilelang_copy2d.py

```py
"""问题 7.4：TileLang 版二维缩放拷贝（填空）。

Y = 2 * X，X 形状 (M, N)，M、N 都不保证整除 tile 边长 (类似 prob 2.6 )
。这次把 tile 先搬进 shared memory，算完再写回：数据搬运交给
T.copy.
填完对照 2.6 想一想：行列号、边界保护、grid 尺寸这几个空，
哪些在这里还有对应，哪些被 T.copy 吃掉了。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k copy2d
"""


import tilelang
import tilelang.language as T


def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale2d(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid，和 7.3 一样——x 方向管 N 列，
        #         y 方向管 M 行，提示：T.ceildiv ======
        
        block_x = T.ceildiv(N, block_N)
        block_y = T.ceildiv(M, block_M)
        
        with T.Kernel(block_x, block_y, threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)

            # ====== 空 2：把当前 tile 从 X 搬进 shared。
            #         提示：T.copy(X[行起点, 列起点], X_shared)，
            #         越界部分 T.copy 会自己处理 ======
            start_x = bx * block_N
            start_y = by * block_M
            
            T.copy(X[start_y,start_x], X_shared)

            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0

            # ====== 空 3：把算完的 tile 写回 Y 的同一位置 ======
            
            T.copy(X_shared, Y[start_y, start_x])

    return scale2d
```

# prob 7.5 （CONCEPT）

补全下表，每个空填“用户”或“编译器”（二者都涉及的要写清楚各自的范围）

| 谁负责                | CUDA SIMT | cuTile  | Triton  | TileLang                |
| ------------------ | --------- | ------- | ------- | ----------------------- |
| 线程到数据的映射| 用户   | **编译器** | **编译器** | **编译器**       |
| 边界处理  | 用户  | **编译器** | **用户**  | **二者都有**      |
| tile / block 尺寸的选择 | 用户  | 用户      | 用户      | 用户     |
| block 内同步   | 用户   | **编译器** | **编译器** | **编译器为主；低层写法也可由用户显式控制** |

# probably7.6 (Fill-in)

请填空：kernels/tilelang_matmul.py ，具体要求见代码注释。

```py
"""问题 7.6：TileLang tiled matmul（填空）。

C = A @ B，A 形状 (M, K)，B 形状 (K, N)，fp16 输入、fp32 累加。
共五个空：两块 shared tile、一个 fragment 累加器、沿 K 维的流水
循环、T.copy 搬运与 T.gemm 计算。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k matmul

Bonus 用的也是这个文件：填完后调 bench() 里的配置，记录实测数据。

    python -c "from kernels.tilelang_matmul import bench; bench()"
"""

def make_matmul(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                threads=128, num_stages=3,
                dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=threads,
        ) as (bx, by):
            # ====== 空 1：A、B 各自的 shared tile——形状分别是多少？ ======
            A_shared = T.alloc_shared((BLOCK_M,BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K,BLOCK_N), dtype)

            # ====== 空 2：C 的累加器 tile，放寄存器（fragment），
            #         注意精度用 accum_dtype ======
            C_local = T.alloc_fragment((BLOCK_M,BLOCK_N), accum_dtype)

            T.clear(C_local)

            # ====== 空 3：沿 K 维流水地推进——一共要多少步？提示：T.ceildiv ======
            for k in T.Pipelined(T.ceildiv(K,BLOCK_K), num_stages=num_stages):
                # ====== 空 4：把 A、B 的当前 tile 搬进 shared——
                #         各自的全局起点坐标是多少？ ======
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                # ====== 空 5：tile 级乘累加，提示：T.gemm ======
                T.gemm(A_shared,B_shared,C_local)

            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return main
```


A_shared 和 B_shared 用来边界处理和分小块。然后分成小块分别计算，累加到C对应的位置里面。

# prob 7.7（FROM-SCRATCH）：softmax in TileLang

在kernels/tilelang_softmax.py 里从零实现行 softmax，具体要求见文件开头的docstring

```py
"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T


def make_softmax(M, N , BLOCK_M = 1 , threads = 128 , dtype = "float32",num_stages = 3):

    def next_power_of_N(N):
      if N <= 0:
        return 1
      
      N -= 1
      N |= N >> 1
      N |= N >> 2
      N |= N >> 4
      N |= N >> 8
      N |= N >> 16
      
      return N+1

    BLOCK_N = next_power_of_N(N)
    
    @T.prim_func
    def main(
      A: T.Buffer((M,N),dtype),
      B: T.Buffer((M,N),dtype),
    ):
      with T.Kernel(
        T.ceildiv(N,BLOCK_N),
        T.ceildiv(M,BLOCK_M),
        threads= threads,
      ) as (bx,by):
        
        row = T.alloc_fragment((1, BLOCK_N), dtype)
        max_buf = T.alloc_fragment((1,), dtype)
        sum_buf = T.alloc_fragment((1,), dtype)
        
        for j in T.Parallel(BLOCK_N):
          row[0, j] = T.if_then_else(
                                      j < N,
                                      A[by, j],
                                      -T.infinity(dtype),
                                  )
        
        T.reduce_max(row,max_buf)
        
        for j in T.Parallel(BLOCK_N):
          row[0,j] = T.exp(row[0,j] - max_buf[0])
        
        T.reduce_sum(row,sum_buf)
        
        for j in T.Parallel(BLOCK_N):
          B[by,j] = row[0,j] / sum_buf[0]

    return main

def softmax(x: torch.Tensor) -> torch.Tensor:
    
    M:int
    N:int
    M= x.shape[0]
    N = x.shape[1]
    
    b = torch.randn((M, N), device="cuda", dtype=torch.float32)

    func = make_softmax(M, N)
    kernel = tilelang.compile(func, out_idx=[1])
    result = kernel(x)
    return result
```

```
一个 CTA → 一行
        ↓
将行扩展到 BLOCK_N = next_power_of_2(N)
        ↓
真实元素 | -inf padding
        ↓
reduce_max
        ↓
exp(x - max)
        ↓
reduce_sum
        ↓
normalize
        ↓
写回输出
```


# Prob 7.8 (optional)

```py

"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl

@triton.jit
def soft_max_kernel(
  x_ptr,
  y_ptr,
  N,
  BLOCK_SIZE: tl.constexpr,
):
  row = tl.program_id(0)
  
  row_start = row*N
  
  offsets =  tl.arange(0,BLOCK_SIZE)
  
  mask = offsets < N
  
  x = tl.load(x_ptr + offsets + row_start, mask= mask, other= -float('inf'))
  
  x_max = tl.max(x,axis = 0)
  
  x = x -  x_max
  
  x_exp = tl.exp(x)
  
  x_exp_sum = tl.sum(x_exp, axis = 0)
  
  result = x_exp / x_exp_sum
  
  tl.store(y_ptr + offsets + row_start, result , mask= mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M,N = x.shape
    
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    
    soft_max_kernel[grid](
      x,
      y,
      N,
      BLOCK_SIZE=BLOCK_SIZE
    )
    return y

```

不得不提，Triton确实比tilelang好写不少.
