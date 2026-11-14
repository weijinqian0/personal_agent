# GLM-5.3版本化条件算例

这是2026-11-14讨论的静态算例，不是上板结果或模型通用常数。部署文档验证于v0.23.0，但分析使用本地vllm-ascend HEAD `72c40a3b097b3989d1aff824cbd380d37d2878ac`和另行读取的op-plugin master包装器，未证明二进制版本一致。

## 当前输入与来源

- 仓库入口：`docs/source/tutorials/models/GLM5.3.md`、`tests/e2e/cases/models/configs/GLM/GLM5_3-W8A8-A3-dual-nodes.yaml`。
- 原文档A3示例为TP4/DP8、单DP4096、fused MC2。**用户后续明确要求EP16、单DP8192**；本算例选TP4/DP4，属于更改后的分析场景，不能称为原文档已验证部署。若TP2/DP8同为EP16，全部容量要重算。
- [实际量化模型config](https://modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8/resolve/master/config.json)：H=6144、I_expert=2048、路由专家256、top-k=8、共享专家1、78层、q_lora=2048、kv_lora=512、64 query heads、qk_nope=192、rope=64、V维256、index_topk=2048。config文件revision为`7425e1e1982dc8dd640fd24a36772d258e1a549f`。
- 实际quant_model_description中路由与共享专家为W8A8_DYNAMIC，router为FLOAT；元数据文件revision为`1076c5d65799cd0840c4df3424fba4a2db864dd6`。不能拿官方FP8 checkpoint代替该量化路径。
- DSA-CP、FlashComm1、FULL_DECODE_ONLY、max_num_seqs=6、MTP推测token=3/draft eager。比较时只固定prefill为AG/A2AV，decode路径保持一致；未改服务，也未验证改后配置能装下权重。
- 每DP8192，T_unique=8192×4=32768；SP源rank token=8192/4=2048；N_recv=32768×8/16=16384；每卡16个专家、每专家1024对。专家完整EP放置，I不再除TP4。

## MoE主要张量：分配量不等于有效量

AG的SP W8A8 prepare先量化再gather：hidden `[32768,6144]` INT8=192MiB，router `[32768,256]` FP32=32MiB。

所查路由包装器按`min(active_num,T×k)`分配，此处active_num=262144，active_expert_range只约束有效计算。路由输出`[262144,6144]` INT8=1536MiB。GMM1 adapter以x.size(0)分配INT8 `[262144,2048]`=512MiB；GMM2也沿用行数，BF16 `[262144,6144]`=3072MiB。

AG主要共存峰值候选：`192+32+512+3072=3808MiB=3.71875GiB`，在GMM2输出分配。需核实目标TorchNPU版本确实沿用这些行数；compact实现会不同。

A2AV按16384对收发，输入INT8=96MiB、激活INT8=32MiB、输出BF16=192MiB。combine时上层仍持有旧mlp_output，逆重排分配发送、通信再分配接收，三份输出共存：

`3×16384×6144×2 = 603979776 bytes = 576MiB = 0.5625GiB`。

两者都是主要张量候选，未包含相同边界下的全部hidden/residual/共享buffer、算子工作区和其他阶段，不是完整峰值或完整峰值差。

## Attention：C8 Query拼接副本不能漏

DSA-CP本卡query为2048行、完整64头。Sparse Attention输出分配时可能共存：

| storage | MiB |
|---|---:|
| ql_nope `[2048,64,512]` BF16 |128|
| q_pe `[2048,64,64]` BF16 |16|
| 新cat query `[2048,64,576]` BF16 |144|
| latent输出 `[2048,64,512]` BF16 |128|
| 本轮top-k `[2048,1,2048]` INT32 |16|
| qkv_lora `[2048,2624]` BF16 |10.25|
| normalized q_c `[2048,2048]` BF16 |8|
| 完整hidden输入backing、wrapper预分配输出各96 |192|
| 模型共享top-k `[8192,2048]` INT32 |64|
| residual，分片/完整布局 |24或96|

按native C8/上述布局，QSFA tiling基础workspace为`4.1298828125MiB×AIC数+平台libapi workspace`；Indexer为`2.251953125MiB×AIC数+平台libapi workspace`。基于TND、mBase128、sInner512、headDimAlign576、preload2、未启用split-KV。不同workspace版本重新推导。

**条件示例**：假设AIC=24、平台libapi workspace=16MiB（未在设备确认），QSFA workspace=115.1171875MiB。加入表中共存项得到845.3671875或917.3671875MiB，即约0.826或0.896GiB。24核/16MiB不是硬件默认事实，两个值也不是上下界保证。

仍缺KV写入/gather staging、RoPE表、metadata、额外executor workspace和allocator开销。V投影的64MiB输出发生在cat退出后，不能再把它与cat峰值机械相加。Attention候选含公共hidden和索引，而A2AV 0.5625只含三份输出，不能直接对比宣布Attention更大。

## HCCL与图池：未知不能变成零

当前源码普通域options=200、DP最小50、MC2沿用HCCL_BUFFSIZE=400。若TP/DP/EP/MC2四个独立域都初始化、各两份收发且按MiB换算：

`2×(200+50+200+400)MiB = 1700MiB = 1.66015625GiB`。

这是未经完整通信域/版本审计的条件算例，缺world等其他域、复用/惰性行为和链接/context开销。不能说“整卡HCCL已确认1.66GiB”。

AG主要张量加此通信例=5.37890625GiB，A2AV=2.22265625GiB，Attention示例=约2.486～2.556GiB。各自边界仍不完整，均不能作为部署预算。

FULL_DECODE_ONLY仍保留decode图池G。无capture日志不能赋值；如果non_torch已包含通信，则不能重复再加1.66GiB。完整总额按同边界阶段max和storage并集计算。

## 复用前核对

确认量化模型、后端/算子版本、分配行数、SP/DSA-CP、专家并行、token budget、dtype、融合和capture路径全部匹配。任一不符，重新取证与复算；不通过调大常数掩盖遗漏。
