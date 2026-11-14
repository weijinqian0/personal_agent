# 版本化算例：DeepSeek V4.1 Flash

此案例用于检查核算过程，不是新的部署输入或默认参数。读取本例后仍须确认用户部署、源码 revision 和工作区修改。

2026-11-14 核对的本地代码：

- vLLM HEAD `59e6682c21595b3f71c4a569318d1a1ac328add3`。
- vLLM-Ascend HEAD `72c40a3b097b3989d1aff824cbd380d37d2878ac`，其中 V4.1 model.py 有未提交修改：router 改 GateLinear、关闭 precast_fp32_weight。
- 固定 BF16 model dtype、原生 FP8 Linear/packed FP4 routed experts，TP2、有效 EP32（DP16×TP2）、专家内部 TP1、PP/PCP/DCP1、Engram 大表 offload；无 mix_placement、冗余专家、视觉和 DSpark/MTP。

在当前修改后的工作区，关闭 DSA-CP、开启共享专家 DP，静态加载后分项为：

| 模块 | GB/卡 |
| --- | ---: |
| 路由专家及 scale | 9.024307200 |
| 共享专家及展开后的 scale | 1.459814400 |
| 主注意力与 compressor | 2.868990976 |
| Indexer | 0.046400512 |
| Embedding 与 LM head | 1.323827200 |
| Router/norm/mHC 等 | 0.315533760 |
| Engram 投影/gate 及 scale | 0.324567040 |
| 合计 | 15.363441088 |

该快照的条件变化：

- 共享专家 DP 关闭，恢复 TP2：减少0.7299072 GB/卡。决定因素是 Ascend 底层 get_parallel_op 的共享专家专用开关，不是 SP。
- DSA-CP 开启：wq_b 及 sink 复制，增加0.86508032 GB/卡。
- A5 full-O-proj 路径还使用约0.077856768 GB/卡共享池；同形状层共享，不乘40层。
- 恢复已提交 HEAD 的 precast：原BF16 router保留，额外FP32副本增加0.3145728 GB/卡；不能只算dtype净差。
- 本地原生DS线性量化保留FP8 wo_a；LM head保留BF16。不能套用官方简化参考实现的BF16 wo_a/FP32 head。
- FP8线性的32×32 block scale展开为每输出、每输入group32一个uint8 scale；本基线scale增量0.13586432 GB/卡。
- ratio2 compressor升为FP32，ratio1不升，额外0.03145728 GB/卡。
- Engram host/UVA大表没有整表HBM副本；投影/gate保留。两个层的BF16 lookup缓冲合计24576 bytes/capacity token；capacity8192时为192MiB，属于非权重缓冲。
- 开启视觉塔还需约0.97 GB/卡（本地BF16配置）；只有实际禁止加载视觉塔才可排除，仅输入纯文本不足以排除。

历史源码定位：Ascend `models/deepseek_v41/{model,indexer,compressor}.py`、`models/deepseek_v41/engram/{embedding,npu,layer}.py`、`ops/{linear,linear_op}.py`、`quantization/methods/w8a8/w8a8_mxfp8.py`（ds_linear）、`quantization/methods/w4a8/w4a8_mxfp4.py`（ds_w4a8_moe）、`attention/context_parallel/dsa_cp.py`、`weight_switch/linear.py`；上游 `model_executor/layers/fused_moe/config.py`。

若本地存在 ai-infra-book，原始记录与header驱动的可复算脚本位于 `research/v41-weight-hbm-code-audit-2026-11-14/`。脚本是该快照的固定场景复算，不能直接用于其他版本/模型；独立安装的skill不依赖该仓库。数字仅覆盖静态张量存储，未测量CANN NZ物理padding、allocator和运行时峰值。
