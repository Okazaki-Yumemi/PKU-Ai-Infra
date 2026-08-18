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