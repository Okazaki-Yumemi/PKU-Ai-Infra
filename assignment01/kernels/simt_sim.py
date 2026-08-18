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
    
    
    
    
    
    
        
        