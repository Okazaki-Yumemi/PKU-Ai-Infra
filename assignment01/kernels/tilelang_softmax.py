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