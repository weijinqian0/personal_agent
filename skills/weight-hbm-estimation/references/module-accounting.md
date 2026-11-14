# 模块与代码核对指南

本表给出检索方向，不是所有框架都成立的布局默认值。W 指该模块完整运行格式字节，t 指该模块实际 TP，e 指有效 EP。

| 模块 | 候选布局 | 必查项 |
| --- | --- | --- |
| Token embedding / LM head | 词表切分 W/t；或完整复制 | 独立词表组、词表 padding、model dtype、量化配置是否支持 embedding/head、PP 首尾和权重绑定的实际 storage |
| 普通 Q / 输出投影 | 按头/输入维切分 W/t | 底层组、头数整除、输出归约路径 |
| GQA K/V 投影 | 按 KV 头切分或复制 | TP 超过 KV 头数后的 local head 数；权重复制与 KV cache 复制分别计量 |
| MLA Q 低秩下投影、KV 公共下投影 | 常见完整复制 | ReplicatedLinear 是否被插件替换、低秩空间是否另有切分 |
| MLA Q/KV 上投影、输出投影 | 常见按头切分 | absorption 的实际执行/保存布局、融合矩阵是否额外保留；不能从 KV 压缩比例推权重比例 |
| V4 的 wq_b/wo_a/wo_b | 常见按头/组切分 | DSA-CP 的 wq_b 复制、独立 O TP、full-O-proj gather 池、池是否跨层共享 |
| attn_sink / per-head 参数 | 随实际本地头或完整复制 | CP 与普通路径的参数长度 |
| Indexer Q、weights_proj | 完整复制或按头切分 | 看具体引擎；不要沿用官方参考实现。计入共享低秩 Q 的归属，避免重复算 wq_a |
| Indexer K、norm | 公共复制或仅 owner 层持有 | owners/reindex/reuse 各类层实例化哪些参数；状态共享不等于权重共享 |
| Compressor wkv/wgate、norm、APE | 常见复制 | ratio 对 dtype 的分支、owner 层数、FP32 常驻还是仅临时计算 |
| Dense FFN gate/up/down | 常见中间维 TP | 融合 gate_up 的两个逻辑矩阵、MLP 独立 TP、padding |
| Router、纠偏 bias、hash table | 常见复制 | BF16 与 FP32 是否同时常驻、hash 表是否真实存在、int dtype、用户未提交改动 |
| 路由专家 | EP 按专家分配；可另有专家内部 TP | 有效 EP 与 CLI DP 的关系、本地物理专家数、冗余/热点副本、packed dtype 与 scale |
| 共享专家 | DP 完整复制 W；或 TP W/t | 专用 shared_expert_dp 开关、底层是否覆盖 disable_tp、SP 不能直接等同共享专家 DP、mix_placement 是否迁入专家池 |
| 公共潜空间进出投影 | 复制或独立切分 | 是否属于单专家；发送前/接收后位置；不能直接除 EP |
| Norm、mHC/AttnRes 等连接参数 | 常见复制 | 可训练参数、执行 buffer 与 FP32 临时张量分别算 |
| Engram 大表及 scale | offload；或表分片 | 有效 cpu_offload、跨 DP 表切分与节点共享、host/UVA、硬件相关 INT8/FP32 或 FP8/E8M0 格式 |
| Engram 投影与 gate | 常见设备端复制 | 大表 offload 不等于这些模块也卸载；设备 lookup 缓冲按 token capacity 另列 |
| 视觉、MTP、DSpark | 按实际模块布局 | 加载开关和模型 wrapper；只发纯文本请求不证明视觉塔没加载；共享 head/embed 按物理 storage 去重 |

## 核算公式与防误用

通用定义优先于“总权重/卡数”：

```text
W_rank = Σ 本卡实际拥有的唯一常驻权重 storage
       + Σ 对应量化元数据 storage
       + Σ 额外常驻权重副本/独占执行打包 storage
W_max  = max_rank(W_rank)
```

`W/t` 或 `W/(e*t_expert)` 只在对应维度独立、均匀、可整除且没有复制时成立。有效 EP 若已包含 Attention TP，就不能再用 Attention TP 除专家。PP 按本阶段实际层与首尾参数分配，不能不加判断地整体除 PP。

量化载荷：FP16/BF16 为2 bytes/element；FP8/INT8 为1；packed FP4/INT4 通常两参数一字节。最终以存储 shape 与 dtype 为准，避免对已经 packed 的 shape 再除2。scale 按运行时 shape×element_size；block scale 转 per-channel/group scale 可能增大。

`format_cast` 可能增加物理 padding；仅凭 tensor.numel 不一定得到 NPU storage bytes。区别旧参数被替换、旧参数保留、别名、缓存分配器保留与临时转换峰值。无法从有效算子实现确认、又影响用户所需结果时，询问硬件/运行时版本或请求对应实测数据，不默填。

## 检索路线

先用 `rg --files` 找模型实现、quant config、Linear/FusedMoE、并行配置及注册/patch 文件，再沿调用链检索：

```text
get_quant_method / register_quantization_config / register_model / patch
get_parallel_op / get_replicated_op / disable_tp / get_*_group
use_sequence_parallel_moe / enable_shared_expert_dp / mix_placement
FusedMoEParallelConfig / ep_size / num_redundant_experts
create_weights / get_weight / weight_loader / process_weights_after_loading
precast_fp32_weight / weight_fp32 / clone / contiguous / format_cast
weight_scale / repeat_interleave / group_size
cpu_offload / HostUvaBuffer / SharedUvaBuffer / allocate_weights
weight_switch / full_pools / pool_key / max_num_batched_tokens
```

源码路径随版本迁移；找不到某个历史路径时按符号重搜。对非 Ascend 场景采用对应后端代码；不能用另一后端填补 Ascend 的未知项。
