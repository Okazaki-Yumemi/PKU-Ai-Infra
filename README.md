# AI Infrastructure Seminars

> Personal learning repository for **LCPU / PKU AI Infrastructure Seminars**.

This repository records my implementations, experiments, debugging notes, and performance analysis for the AI Infrastructure seminar series jointly organized by the **PKU Supercomputing Team** and **PKU Linux Club (LCPU)**.

The goal is not only to finish the assignments, but also to systematically build practical understanding of GPU programming, kernel optimization, and modern AI systems infrastructure.

## Course

* Course: **AI Infrastructure Seminars**
* Official Website: https://infra.seminars.lcpu.dev/
* Official Repository: https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars
* Topics include:

  * GPU architecture and programming
  * CUDA
  * Triton
  * TileLang
  * Kernel optimization
  * Profiling and performance analysis
  * AI infrastructure systems

## Repository Purpose

This is an **independent personal repository**, rather than a GitHub fork of the official course repository.

The official repository is maintained locally as:

```text
upstream -> lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars
origin   -> this repository
```

Course updates are fetched from `upstream`, while my own implementations, experiments, and notes are pushed to `origin`.

Typical synchronization workflow:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## Development Environment

### Local

```text
OS          Windows + WSL Ubuntu
GPU         NVIDIA RTX 5070 Laptop GPU
CUDA CC     sm_120
VRAM        ~8 GB
Editor      VS Code
Python      uv-managed environment
```

### Final Evaluation

```text
GPU         NVIDIA B300
```

Local benchmark results should therefore **not** be interpreted as final performance conclusions.

In particular:

* correctness should remain portable across GPUs;
* hardware-specific parameters should not be unnecessarily hard-coded;
* tile size, block size, number of warps, shared-memory usage, etc. may require retuning;
* performance measurements on the RTX 5070 do not necessarily predict performance on B300.

## Workflow

For each kernel or assignment, I try to follow the workflow:

```text
Understand the computation
        ↓
Build a reference implementation
        ↓
Verify correctness
        ↓
Analyze execution / memory mapping
        ↓
Establish a performance model
        ↓
Profile
        ↓
Optimize
        ↓
Re-verify correctness
        ↓
Benchmark again
```

Correctness is treated as a prerequisite for optimization.

Typical correctness checks include:

* small input sizes;
* random shapes;
* non-aligned / non-divisible shapes;
* boundary cases;
* multiple data types;
* comparison against PyTorch / CPU reference implementations.

## Performance Analysis

Kernel optimization is guided by architectural reasoning rather than benchmark-only trial and error.

Main aspects include:

### Memory

* global memory traffic
* coalescing
* cache behavior
* shared memory
* register usage
* redundant memory accesses
* arithmetic intensity

### Parallelism

* thread / warp / block mapping
* warp utilization
* occupancy
* latency hiding
* load balancing

### Compute

* instruction count
* vectorization
* instruction-level parallelism
* Tensor Core utilization

### Synchronization

* block barriers
* warp synchronization
* unnecessary synchronization

When appropriate, I use a Roofline-style model to determine whether a kernel is primarily:

```text
memory-bound
compute-bound
latency-bound
launch-overhead-bound
```

## Benchmark Methodology

GPU execution is asynchronous, so benchmark methodology is considered part of the implementation.

Measurements should account for:

* warmup;
* CUDA synchronization;
* repeated execution;
* measurement variance;
* median / mean latency;
* cache effects;
* JIT compilation overhead.

For CUDA/Triton/TileLang code, I distinguish:

```text
compile / JIT time
```

from:

```text
steady-state kernel execution time
```

The first execution is therefore generally not used directly as kernel latency.

## Assignment Notes

For important debugging or optimization results, I try to keep notes in the following format:

```text
Problem:
Root cause:
GPU concept:
Fix:
Performance impact:
Hardware dependency:
```

The purpose is to preserve not only the final implementation, but also the reasoning process that led to it.

## Current Focus

### Assignment 1 — GPU & GPU Programming

Main topics:

* CUDA execution model
* threads / warps / blocks
* memory hierarchy
* coalesced memory access
* synchronization
* shared memory
* GPU reductions
* Triton programming model
* TileLang
* kernel profiling
* GPU performance optimization

Local development is performed primarily on the RTX 5070 Laptop GPU, with final validation and performance evaluation intended for the B300 environment.

## Academic Integrity

This repository is primarily intended for personal learning and experimentation.

Implementations are developed with the goal of understanding the underlying GPU mechanisms rather than exploiting benchmark-specific behavior.

In particular, optimizations should avoid:

* hard-coded benchmark outputs;
* caching expected answers;
* bypassing required computation;
* exploiting evaluation framework bugs;
* unjustified fixed-shape special cases.

Shape specialization is acceptable only when it is a legitimate optimization implied by the problem or workload.

## Goal

By the end of the seminar series, I hope to be able to independently:

* read and understand CUDA kernels;
* design thread/block mappings;
* reason about GPU memory access patterns;
* identify likely kernel bottlenecks;
* use profiling tools to validate performance hypotheses;
* optimize kernels while maintaining correctness;
* understand how Triton and TileLang abstract CUDA execution;
* transfer kernel implementations across different GPU architectures.

The emphasis of this repository is therefore:

> **correctness, architectural understanding, profiling, and explainable optimization.**
