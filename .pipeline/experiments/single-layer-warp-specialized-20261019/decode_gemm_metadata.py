"""Decode public descriptor enums on CPU; do not infer actual kernel counters."""
import hashlib
import json
from pathlib import Path
import re

B = Path(__file__).resolve().parent
HEADER = Path('/rjs/cxz/miniconda3/envs/va/lib/python3.12/site-packages/nvidia/cublas/include/cublasLt.h')


def main():
    source = HEADER.read_text()
    tiles = {int(i): (int(a), int(b)) for a, b, i in re.findall(
        r'CUBLASLT_MATMUL_TILE_(\d+)x(\d+)\s*=\s*(\d+)', source)}
    stages = {int(i): (int(a), int(b)) for a, b, i in re.findall(
        r'CUBLASLT_MATMUL_STAGES_(\d+)x(\d+)\s*=\s*(\d+)', source)}
    data = json.loads((B / 'local-gemm-inspect-1/results/RESULT.json').read_text())
    rows = []
    for r in data['rows']:
        cfg = r['configuration']
        a, z = tiles[cfg['tile_id']]
        size, number = stages[cfg['stages_id']]
        rows.append(dict(batch=r['batch'], name=r['name'], logical_M=r['M'],
            logical_N=r['N'], logical_K=r['K'], tile_id=cfg['tile_id'],
            enum_name=f'CUBLASLT_MATMUL_TILE_{a}x{z}', logical_tile_M=z, logical_tile_N=a,
            stage_id=cfg['stages_id'], stage_enum=f'CUBLASLT_MATMUL_STAGES_{size}x{number}',
            shared_stage_size=size, shared_stage_count=number,
            splitK=cfg['splitK'], reduction_scheme=cfg['reduction_scheme'],
            actual_CTA_count='UNKNOWN; not inferred from tile enum', source_interception=False))
    result = dict(header_source=str(HEADER), header_SHA256=hashlib.sha256(HEADER.read_bytes()).hexdigest(),
        library_version=data['rows'][0]['library_version'], rows=rows,
        observed_on='local A100-SXM4 GPU-754f96ec-4567-7af2-7048-7e8957a6eb36',
        remote_heuristic_identity='UNKNOWN; not independently queried or intercepted',
        orientation='gemm.h uses column-major Y.T=W X.T; reverse vendor descriptor m/n for logical output M/N',
        heuristic_reconstruction_only=True, new_GPU_calls=0)
    (B / 'CUBLASLT-TILE-ENUMS.json').write_text(json.dumps(result, indent=2) + '\n')
    text = [
        '# 原型与成熟原语：可核查的差异和下一步', '',
        'OBSERVED：整个一层单kernel和warp分工已实现，但这只是执行结构；原数值与速度未通过。主源码维持固定，下面只读解码已有8条公共API metadata，没有新GPU调用。', '',
        '|B/原语|逻辑M/N/K|公共tile换回逻辑M×N|shared stage枚举size×count|splitK|',
        '|---|---|---|---|---:|',
    ]
    for r in rows:
        text.append(f'|B{r["batch"]}/{r["name"]}|{r["logical_M"]}/{r["logical_N"]}/{r["logical_K"]}|'
                    f'{r["logical_tile_M"]}×{r["logical_tile_N"]}|'
                    f'{r["shared_stage_size"]}×{r["shared_stage_count"]}|{r["splitK"]}|')
    text += [
        '', '这是本机A100上固定descriptor的heuristic重建，不是已结束Torch调用的现场拦截；远端heuristic身份未另查询，UNKNOWN。tile方向依据项目gemm.h的Y转置列主序；实际CTA数、指令顺序、L2/DRAM和achieved occupancy均不能由枚举推断。头文件原位置/SHA见CUBLASLT-TILE-ENUMS.json，没有复制或修改库源码。', '',
        'OBSERVED：新原型所有GEMM统一64×64×32、单缓冲同步load→MMA→store；重建原语则按形状选择不同tile和3–5级shared staging。B14 FFN1与cross K/V等大M工作没有获得成熟库同等的tile/流水实现。增加候选本来已经让原库M32→224，不能把库已获得的收益算给新队列。', '',
        '容量推导：A100 SM80每SM 65536个32-bit寄存器、最多64 warp。新函数172寄存器/线程×256线程=44032，尚未计分配取整就排除两个CTA同时驻留；实际API也返回1 CTA/SM。因此最多8驻留warp/SM，即12.5%的理论warp容量，不是实测achieved occupancy。仅4个warp承担主要数学工作；加载/写回阶段没有实现真正重叠时，其余角色会等待。[NVIDIA CUDA12.4.1 Ampere指南](https://docs.nvidia.com/cuda/archive/12.4.1/ampere-tuning-guide/index.html)。', '',
        'OBSERVED：三B14的GEMM阶段合计74.66%，cross K/V投影占49.2%–63.1%；ldm40唯一反事实下降6.5%–19.0%，但仍原精度FAIL。HYPOTHESIS：同步搬运、粗grid barrier、单warp写回、较小tile与寄存器资源共同限制吞吐；本轮没有细到每个stall的硬件计数，不能把其中一个指认为唯一根因。', '',
        '与MegaMoE的差别：已借鉴任务领取和warp职责，但没有SM100的TMA/UMMA/TMEM或动态寄存器能力，也没有其ready驱动跨阶段生产者消费者流水。当前是29个粗阶段的cooperative执行器，中间量写global；让所有算子同处一个launch没有消除这些数据依赖与中间流量。', '',
        '唯一默认后继建议：先分别解决同输入Q、FFN2及attention的设备数值边界，再将一个已经通过的成熟SM80 tile原语接回整层。随后只选一处有价值的load/compute双缓冲或ready子图，明确generation、发布/获取、消费者释放与无死锁条件。CUDA12.4.1官方支持Ampere异步global→shared和arrive/wait barrier；不需要升级到CUDA13，也不把其他线程的wait当成producer复制完成。', '',
        '强baseline必须双方都使用已有cached/shared-prefix/Graph；当前昂贵cross K/V在成熟缓存命中时本来消失，不能通过新增普通缓存宣称调度创新。先守住原算术/状态，再判断tile流水是否超过强B14。当前没有新的GPU作业、没有生产集成，不扩30层或环境，不开启register/tile网格搜索。',
    ]
    (B / 'IMPLEMENTATION-ISSUES.md').write_text('\n'.join(text) + '\n')
    print('decoded 8 descriptors; zero new GPU calls')


if __name__ == '__main__':
    main()
