# W.1 Parallel Programming with TileLang — 精简阅读笔记

> 基于课程转录与 Workshop slides 整理。目标不是复刻 86 分钟讲课，而是保留对后续 Assignment / Tensor Core / GPU Kernel 优化真正有用的主线。
>
> 建议阅读时间：15–25 分钟。今天状态一般时，只看每节的 **一句话结论** 和最后的 **速查表** 即可。

---

## 0. 这节 Workshop 到底在讲什么？

这节课的核心并不是“教几个 TileLang API”，而是在回答：

> **写一个 GPU kernel 时，哪些事情由程序员决定，哪些事情可以交给 TileLang / compiler？**

从 architecture 视角，写 kernel 主要要处理四个问题：

1. **Work Partition**：CTA / warp / thread 各自负责哪些逻辑元素？
2. **Memory Placement**：数据放在 global / shared / register(fragment) 哪一级？
3. **Scheduling**：load / compute / store 如何排序，怎样 overlap？
4. **Tuning**：tile size、threads、stages、worker CTA 数量该怎么选？

TileLang 的定位不是“把 GPU 编程自动化”，而是：

- 程序员仍然决定 **CTA-level workload decomposition**；
- 程序员仍然决定主要的 **memory scope**；
- 程序员仍然决定总体的 **pipeline / task structure**；
- 编译器帮助推导大量 **tile 内部的 thread ownership、register layout、copy lowering、target-specific implementation**。

可以把 TileLang 理解为：

```text
你写：            compiler 帮你：
-----------------------------------------
CTA 算哪块 tile    tile 内元素怎么分给 threads
数据放哪一级        fragment / shared 的部分物理映射
T.copy / T.gemm    具体 lower 成什么指令和访问
pipeline 结构       prologue / steady / epilogue 等展开
tile/stages 参数    不负责替你自动找到普遍最优参数
```

---

# 1. 从 CUDA 到 TileLang：抽象层发生了什么变化？

CUDA 更偏 **thread-centric**：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

你显式思考：

> “这个 thread 处理哪个元素？”

TileLang 更偏 **tile / CTA-centric**：

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

你先思考：

> “这个 CTA 处理哪一块数据？”

然后在 tile 内使用：

```python
T.Parallel(...)
T.copy(...)
T.gemm(...)
```

描述逻辑并行与 tile-level 操作，由 compiler 去推导更多 thread-level 映射。

### 一句话结论

> **TileLang 没有消灭 thread，而是把很多 thread-level bookkeeping 从用户代码里拿走了。**

---

# 2. TileLang 的最小编译心智模型

课程中大量讲了 JIT、TIRx、Lowering、TVM-FFI。现在不需要记 compiler pass 名字，先记住下面这条链：

```text
Python TileLang DSL
        ↓
TIRx / IRModule
        ↓
一系列 compiler passes
        ↓
Layout Inference
        ↓
TileOp Lowering
        ↓
target-specific device code
        ↓
CUDA / CuTeDSL backend
        ↓
cubin / SASS
        ↓
CUDA Driver
        ↓
GPU
```

## 2.1 JIT / specialization

TileLang 支持 JIT。核心是区分：

### 静态参数

会改变 kernel 实现结构，例如：

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_stages
layout
shared-memory allocation
```

这些适合做 compile-time specialization。

### 动态参数

主要影响具体 workload 大小，例如：

```text
token count
batch size
某些 problem size
loop trip count
```

如果它们不改变 kernel 结构，可以尽量保持动态，提高 artifact 复用率。

### 一句话结论

> **会改变“kernel 长什么样”的参数更适合静态特化；只改变“跑多少工作”的参数更适合动态传入。**

---

## 2.2 为什么保留高层 IR 很重要？

如果一开始就把：

```python
T.copy(A_tile, A_shared)
```

展开成：

```text
thread 7 load address ...
thread 11 store address ...
```

那么 compiler 很早就丢掉了“这是一个 tile copy”的语义。

保留 `T.copy` 这样的 TileOp，可以让 compiler 后面根据：

- source / destination scope
- shape
- alignment
- target architecture
- schedule

决定具体 lower 成：

```text
scalar/vector load-store
CTA cooperative copy
cp.async 风格搬运
TMA
其他 target-specific copy path
```

类似地：

```python
T.gemm(A_shared, B_shared, C_local)
```

仍然保留：

- M/N/K tile
- dtype
- operand scope
- GEMM 语义

因此后端才有机会选择 MMA / WGMMA / tcgen05 等路径。

### 一句话结论

> **越晚丢掉 tile-level semantic information，compiler 的优化空间越大。**

---

# 3. `T.Kernel`：你仍然要决定 CTA workload

```python
with T.Kernel(
    T.ceildiv(N, BLOCK_N),
    T.ceildiv(M, BLOCK_M),
    threads=128,
) as (bx, by):
    ...
```

这里程序员仍然决定：

- 一共有多少 logical CTA；
- 一个 CTA 负责哪个 tile；
- tile 有多大；
- 一个 CTA 有多少 threads。

TileLang 不替用户选择数学分块。

## 为什么 tile 不是越大越好？

tile 太小：

- reuse 不足；
- ILP 不足；
- program / CTA 数过多；
- 更难隐藏 latency。

tile 太大：

- register pressure 增大；
- shared-memory footprint 增大；
- occupancy 可能下降；
- 某些配置甚至无法 launch。

这和 Assignment 1 Bonus 里你已经观察到的情况一致：

```text
64×64 可能优于 128×128
```

### 一句话结论

> **compiler 可以帮你解决 tile 内 mapping，但 workload decomposition 仍是用户的性能责任。**

---

# 4. Fragment：CTA 共同持有的“逻辑寄存器 tile”

这是这节课最值得掌握的概念之一。

例如：

```python
C_local = T.alloc_fragment((128, 128), "float32")
```

它不代表：

> 每个 thread 都拥有一个完整的 128×128 数组。

而是：

> CTA 逻辑上共同持有一个 128×128 fragment，这个 fragment 的元素实际分布在 CTA 内各个 thread 的 registers 中。

物理映射可以理解成：

```text
logical index (i, j)
        ↓
fragment layout
        ↓
(thread_id, thread-local register slot)
```

因此：

```text
logical tensor
≠
每个 thread 的完整私有 tensor
```

### Memory scope 对照

| 逻辑对象 | 物理意义 |
|---|---|
| `T.Tensor` / input-output buffer | Global Memory |
| `T.alloc_shared` | CTA shared memory |
| `T.alloc_fragment` | 分布在 CTA 各线程 registers 中的逻辑 tile |

### 一句话结论

> **Fragment 是“CTA 视角的逻辑寄存器块”，而不是“每个线程一个大数组”。**

---

# 5. `T.Parallel`：声明逻辑并行域，不是创建那么多 CUDA threads

比如：

```python
for i, j in T.Parallel(32, 32):
    ...
```

不是：

```text
创建 32×32 = 1024 个 CUDA threads
```

它只是声明：

```text
logical iteration domain = 32×32
```

接下来 compiler 会推导：

```text
(i, j)
  ↓
哪个 thread 执行
  ↓
该 thread 的第几个 local iteration
  ↓
访问哪个 register slot
  ↓
是否形成 vectorized access
```

### 一句话结论

> **`T.Parallel` 描述的是 logical parallelism，而不是 physical thread count。**

---

# 6. Layout Inference：TileLang 最重要的自动化之一

Layout Inference 的任务可以粗略理解为：

> **给定一组 tile-level 约束，求出 logical elements 到 physical GPU resources 的映射。**

它会结合：

- 显式 layout annotation；
- `T.gemm` 等 TileOp 的 ISA contract；
- `T.copy` 的 source / destination；
- `T.Parallel` 的逻辑迭代域；
- fragment ownership；
- shared-memory layout；

进行约束传播。

最终关心的主要映射包括：

## Fragment layout

```text
logical element
→ thread_id + thread-local register slot
```

## Parallel loop layout

```text
logical iteration
→ thread
```

## Shared layout

```text
logical coordinate
→ shared-memory physical address / swizzle
```

可以把它理解为：

```text
logical program
      ↓
Layout Inference
      ↓
physical ownership / mapping
```

### 一句话结论

> **Layout Inference 本质上是在 compiler 里解“GPU 数据映射约束问题”。**

---

# 7. Shared Memory、Bank Conflict 与同步

这一部分大多属于 Assignment 1 已学内容。

## 7.1 Bank conflict 的关键判断

多个 thread：

- 访问不同 bank → 可并行；
- 访问同一 bank 的不同地址 → conflict；
- 访问相同地址 → broadcast / multicast，可不产生 bank conflict。

## 7.2 Transpose 为什么用 shared memory？

Transpose case 中 shared memory 主要有两个作用：

1. **线程间交换 / 数据重排**
2. **配合 padding 改善 bank mapping**

典型数据流：

```text
global input
→ coalesced/vectorized load
→ thread-local micro transpose
→ shared memory
→ sync
→ coalesced/vectorized output
→ global output
```

## 7.3 Layout inference 不等于 synchronization

即使 compiler 知道：

```text
谁写 shared element
谁读 shared element
```

如果是用户手写 shared load/store，producer-consumer race 仍然需要显式 barrier：

```python
T.sync_threads()
```

### 一句话结论

> **Layout 决定“谁拥有/访问什么”，同步决定“什么时候访问才合法”。两者不是同一件事。**

---

# 8. Tile Library Primitives：为什么 `T.copy / T.gemm / T.reduce_*` 有价值？

TileLang 的一个核心思想是：

> 把常见的 tile-level operation 作为高层 IR primitive 保留下来。

例如：

```python
T.copy(...)
T.reduce_sum(...)
T.gemm(...)
```

这样 compiler 仍然知道操作的完整 tile / region 语义。

## `T.copy`

保留：

- source / destination buffer
- memory scope
- region
- direction

允许 compiler 选择合适的 data movement path。

## `T.reduce_sum`

保留：

- reduction region
- reduction dimension
- dtype
- fragment ownership

允许 compiler 推导本地 reduction 和必要的跨线程通信。

## `T.gemm`

保留：

- M/N/K tile
- operand layout
- dtype
- target ISA contract

是 Tensor Core lowering 的重要约束来源。

### 一句话结论

> **这些 primitive 不是“语法糖这么简单”，它们是在 IR 中保留结构化语义。**

---

# 9. Layout Annotation：什么时候需要用户手动规定 layout？

Compiler 能推 layout，但有时算法有明确 ownership 需求。

例如某种 quantization：

```text
连续 8 个元素必须由同一个 thread 持有
```

因为要在线程私有 local buffer 中做 bit packing。

这时可以用 layout annotation 明确告诉 compiler：

```text
这 8 个元素必须归同一个 thread，并映射到连续 local slots
```

### 一句话结论

> **Layout Inference 是自动求解；Layout Annotation 是用户给它额外的强约束。**

---

# 10. Reduction：现在只记 API 选择，不需要读 compiler internals

这部分 transcript 花了很多时间讲 reducer lowering、replication、AllReduce、NamedBarrier 等 implementation detail。

当前阶段只需要保留下面这张决策表：

| partial values 分布 | 常见方式 |
|---|---|
| 全在线程内 | serial / unroll local accumulation |
| 一个 warp 内 | warp reduce |
| fragment / tile reduction | `T.reduce_*` |
| 声明式 partial accumulation | reducer API |
| 跨 CTA | atomic / multi-stage reduction |

关于 `replication / LayoutReducer / FinalizeReducer lowering` 的几十页实现细节，当前可以跳过。

### 一句话结论

> **先根据“partial values 分布在哪里、最终结果谁需要持有”来选择 reduction 方式。**

---

# 11. Warp Scheduler、Latency Hiding 与 Occupancy

GPU 能隐藏 latency 的一个重要原因：

```text
warp A 等 memory
    ↓
scheduler 发射 warp B
    ↓
warp C
    ↓
...
```

resident warp 的上下文已经在片上，不像 CPU process switch 那样需要保存/恢复完整状态。

## Occupancy 是什么？

Occupancy 描述：

```text
当前可 resident 的 warps
/
硬件最大 resident warps
```

它反映的是：

> scheduler 潜在可选择的 work pool 大小。

但要特别注意：

```text
高 occupancy
≠
高 utilization
≠
高性能
```

occupancy 会受：

- registers/thread
- shared memory/block
- threads/block
- architecture limit

限制。

### 一句话结论

> **Occupancy 是 latency hiding 的条件之一，不是最终性能指标。**

---

# 12. `T.Pipelined`：用户可见的软件流水

这是 Workshop 后半段最重要的内容。

逻辑上：

```text
load 0 → compute 0
load 1 → compute 1
load 2 → compute 2
```

Pipeline 希望变成：

```text
time →

load 0
load 1   compute 0
load 2   compute 1
load 3   compute 2
         compute 3
```

核心目标：

```text
用 compute 覆盖 data movement latency
```

TileLang 会把循环改写成类似：

```text
prologue
→ steady state
→ epilogue
```

并处理：

- logical iteration remapping
- multi-version buffer
- rolling buffer
- async producer grouping
- commit / wait
- barrier

---

## 12.1 `num_stages` 为什么不是越大越好？

`num_stages ↑`：

好处：

```text
更多 iteration in-flight
→ producer-consumer 距离拉大
→ latency hiding 潜力提升
```

代价：

```text
shared-memory versions ↑
register live range ↑
resource pressure ↑
occupancy 可能下降
prologue/epilogue 变长
```

因此最优 stages 是一个硬件相关的 tuning 问题。

课程 H100 case 中，stage sweep 显示存在一个中间最优区间，而不是单调越大越快。

### 一句话结论

> **Pipeline depth 是“latency hiding vs resource pressure”的 trade-off。**

---

# 13. Persistent Kernel：和 Pipeline 是两个不同维度

## 普通 grid

```text
logical task 0 → CTA 0
logical task 1 → CTA 1
logical task 2 → CTA 2
...
```

## Persistent

只启动有限数量的长期存活 worker CTA：

```text
worker CTA 0:
task 0 → task 4 → task 8 → ...

worker CTA 1:
task 1 → task 5 → task 9 → ...
```

关键不是“CTA 数一定等于 SM 数”，而是：

> **有限物理 worker CTA 在生命周期内循环处理更大的 logical task domain。**

Persistent 的价值包括：

- 长生命周期的 worker-local state；
- 自定义 task 分配与顺序；
- 减少某些 global atomic；
- 动态 task queue；
- 复杂 producer/consumer；
- irregular workload 调度。

代价：

- 状态 reset 更复杂；
- tail task；
- synchronization；
- queue contention；
- register/shared footprint 增大。

---

## 13.1 Pipeline vs Persistent

这两个概念最重要的区别：

```text
Pipeline
= 时间调度
= 一个 CTA 内，不同 iteration 如何 overlap
```

```text
Persistent
= 空间 / task 调度
= logical tasks 如何映射给有限 worker CTA
```

可以记成：

```text
Persistent = who does the task
Pipeline   = when operations happen
```

两者可以同时出现。

### 一句话结论

> **Persistent 决定 task→CTA；Pipeline 决定 CTA 内 iteration→time。**

---

# 14. MegaKernel：当前阶段只保留一个判断标准

MegaKernel / persistent 风格常见于：

- 多阶段融合；
- 不规则任务；
- long-lived state；
- producer/consumer phase；
- 动态负载均衡。

但课程强调：

> 不应该因为某个 feature “更高级”就使用它。

选择 primitive 时应该从：

```text
data lifetime
dependency
workload regularity
state ownership
profiling evidence
```

出发。

### 一句话结论

> **Persistent / MegaKernel 提供的是更强控制，不是自动性能提升。**

---

# 15. Profiling：优化要形成“证据链”

性能优化不能只凭：

```text
“理论上应该更快”
```

可以依次使用：

## Nsight Compute (NCU)

看单 kernel：

- occupancy
- memory throughput
- stall reason
- pipe utilization
- resource usage

## Nsight Systems (NSYS)

看系统 timeline：

- host-side launch overhead
- CPU/GPU overlap
- kernel timeline
- multi-kernel dependency

## SASS

作为最终 lowering evidence：

- instruction 类型与数量
- async copy
- barrier sequence
- pipeline code structure
- generated implementation 是否真的不同

### 一句话结论

> **优化结论应该由 runtime + profiler + generated code 共同支持。**

---

# 16. 和 Assignment 1 已学内容的对应关系

你已经通过 Assignment 1 覆盖了 Workshop 的不少内容：

| Workshop 内容 | Assignment 1 已学 |
|---|---|
| CUDA grid/block/thread | 已完成 |
| SIMT / warp | 已完成 |
| memory hierarchy | 已完成 |
| shared memory / bank conflict | 已完成 |
| coalescing | 已完成 |
| occupancy | 已实验 |
| CUDA timing / async | 已完成 |
| Triton tile model | 已完成 |
| TileLang `T.Kernel` / `T.Parallel` | 已写过 |
| `T.alloc_shared` / fragment | 已写过 |
| `T.copy` / `T.gemm` | 已写过 |
| `T.Pipelined` | 已使用 |
| PTX / SASS / JIT | 已完成 |

因此这场 Workshop 对你真正新增的核心主要是：

1. **Layout Inference**
2. **Progressive Lowering / TileOp semantic preservation**
3. **Software Pipeline 的 compiler 视角**
4. **Persistent Scheduling**

---

# 17. 最终速查表

## 写一个 TileLang kernel 时先问四个问题

```text
1. Work Partition
   一个 CTA 负责哪块 logical workload？

2. Memory Placement
   数据在哪一级：global/shared/fragment？

3. Scheduling
   load / compute / store 怎么 overlap？

4. Tuning
   tile / threads / stages / workers 取多少？
```

## TileLang 最重要的几个抽象

```text
T.Kernel
→ CTA workload / grid

T.alloc_shared
→ CTA shared-memory tile

T.alloc_fragment
→ CTA 共同持有的 logical register tile

T.Parallel
→ logical parallel iteration domain

Layout Inference
→ logical element / iteration → physical thread/register/shared mapping

T.copy
→ high-level data movement intent

T.gemm
→ high-level GEMM tile intent

T.Pipelined
→ CTA 内 iteration 的时间 overlap

Persistent
→ logical tasks 到有限 worker CTA 的空间调度
```

## 三个不能混淆的概念

```text
Fragment
≠ 每个线程一份完整 tile

T.Parallel(M,N)
≠ 创建 M×N 个 CUDA threads

Layout inference
≠ synchronization
```

## 两个性能 trade-off

```text
更大 tile
→ reuse ↑
→ 但 register/shared pressure ↑、occupancy 可能 ↓

更多 pipeline stages
→ latency hiding ↑
→ 但 resource pressure ↑、occupancy 可能 ↓
```

---

# 18. 阅读完成标准

今天不用继续钻实现细节。

如果你能回答下面五个问题，这个 Workshop 对当前阶段就算完成：

1. TileLang 替用户隐藏了哪些 thread-level 工作？哪些决策仍必须自己做？
2. `T.alloc_fragment((M,N))` 为什么不是每个 thread 一个 M×N 数组？
3. `T.Parallel(M,N)` 为什么不等于 M×N 个 CUDA threads？
4. Layout Inference 在解决什么问题？
5. Pipeline 与 Persistent 的本质区别是什么？

---

## 最短版总结

> TileLang 的核心价值不是“语法更短”，而是让程序员主要在 CTA / tile 层描述 workload、memory placement 和 dataflow，同时保留 `copy/gemm/reduce` 等高层 IR 语义，由 compiler 推导大量 thread-level layout 与 target-specific lowering。  
> 性能上最重要的两条新主线是：`T.Pipelined` 用时间上的 overlap 隐藏 latency；Persistent 用有限长期存活 CTA 对更大的 task domain 做空间调度。最终仍需通过 tuning 与 profiling 判断配置是否真正更快。
