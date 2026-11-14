---
name: kv-cache-budget
description: 根据模型结构、推理后端源码和运行日志估算 KV Cache、窗口与压缩器状态容量，分析 chunk、前缀命中、异步在途、PD 分离及多卡部署的缓存预算差异。适用于 vLLM/vllm-ascend 等推理引擎的容量核算与日志对账。
---

# KV Cache 容量与峰值分析

交付可复算的逐组件、逐请求、逐 rank 缓存账。区分三个结果：长期有效状态字节、请求消耗的物理池块预算、实测持块峰值。另行说明预分配池总量；它不等于请求当前使用量，也不等于整机显存峰值。

## 获取证据

1. 查找用户指定的本地仓库和部署日志，记录模型 config、后端 commit、未提交相关修改及实际启动参数。有多个 checkout 时明确使用哪一个。缺少资料仍可给条件公式，但不制造精确部署数值。
2. 根据模型 config 和实际构造路径列出缓存组件：全局 K/V 或 latent、索引、SWA、主/索引 compressor state、递推/卷积状态、MTP/DSpark。需要结构公式时读取 [Attention 与状态公式](references/attention-formulas.md)。
3. 沿 `get_kv_cache_spec()` 追到 dtype 解析、页字节属性、runner 分配/reshape、缓存分组和调度器回收逻辑。分析 vLLM/vllm-ascend 时读取 [源码查询路径](references/source-tracing.md)，按符号搜索，文件位置不能当成跨版本事实。
4. 从日志核对 group geometry、块数和 registered bytes。若可以取比值，先确认分子分母属于同一 rank、同一池、同一分配范围。记录证据来源为源码、日志、论文或显式假设。

## 建立计算口径

- 长度使用整数 token；明确 128K 是 128000 还是 131072。GB=10^9 bytes，GiB=2^30 bytes。
- 对每类缓存列出：层/owner 数、记录结构、dtype/scale、压缩比、逻辑 block token 数、物理行数、有效页字节、填充页字节、窗口和额外 retention。
- 将历史拆为命中前缀、此前已计算 chunk、本轮新增记录。命中率只有在代表该请求连续 token 前缀时才能换算成长度；服务池平均命中率不是每请求实际命中长度。
- 本地已共享前缀按物理页去重；远端命中是否需要取回由 connector 决定。命中不自动减少当前 query 对旧历史的 Attention，也不自动减少 chunk 的在途 state。
- 从 allocator 明确各 group 的独立 block 消耗和别名关系。共享 backing 的 tensor views 不重复计算；用 pool bytes/block 计费后不再附加同一份 padding。

## 计算调度与峰值

`max_num_batched_tokens` 是同一调度批所有请求的公共预算，不是每请求额度。分别记录实际调度量、实际未结算量和配置保守上界。

当源码支持时：

```text
F_global_max = max_concurrent_batches * max_num_batched_tokens
sum(F_request) <= F_global_max
```

并发批次数从运行版本和 PP/async 配置获取。两批在途不代表两个 forward 同时计算。不得给每条请求重复分配全局 F 上界。

对采用该公式的 sliding-window manager，页数保守预算为：

```text
n = ceil(min(max_model_len, W - 1 + extra_retained_tokens + F) / block_tokens)
    + boundary_slack_blocks
```

边界余量、回收时点、MTP lookahead、prefix checkpoint retention 均从实现取证。这是 admission 上界；不能仅凭 Spec 就证明实际每行 state 都已写入，或所有组件上界同时达到。若实现采用固定 ring/circular state，改用真实固定槽规则，不能套 chunk 线性增长。

PD 分别按 P/D 实际 dtype、并行布局、在途 token 与持块规则核算。交接双端重叠量只有在寿命重叠时相加。输出 token 引入的历史增长按实际喂入模型的位置数计算；首个输出来自 prefill。

## 多请求和多卡

计算逐卡结果前，先从启动参数、最终解析配置、模型构造和 rank 日志提取并行布局；仅缺失或冲突的关键项再向用户补问，不重复询问已有证据。P、D 分别填写：

| 必须确认的布局 | 需要的证据 |
| --- | --- |
| 作用域 | P/D 实例数、每实例设备数；结果按芯片、rank、节点还是整个实例计 |
| Attention TP 与 DP | 实际 Attention TP group、是否启用 DP Attention、请求与 token budget 属于哪个 worker；不能直接套 MoE 的 EP 或启动参数的名义 TP |
| Context parallel | Prefill CP/PCP、Decode CP/DCP 的大小与切分方式；KV、索引、窗口和 state 各自是否切分、复制或归属于特定 rank |
| Pipeline parallel | PP 大小、各 stage 的实际层/owner 分配、MTP/DSpark 放置和影响在途批次数的配置 |
| 缓存组件归属 | 全局 KV、MLA latent、Index K、SWA、compressor/递推状态、共享 owner 的逐 rank shape 和物理副本 |

若这些信息不足以唯一确定逐 rank 的 KV/索引/state 形状、层数或预算作用域，必须明确提醒用户补充 P/D 完整启动命令或上述缺失参数；不得默认为 Attention TP=1、所有43层都在单卡，或将全模型容量直接除以 EP/TP/CP。补充前仍可给全模型逻辑容量及标明假设的候选场景，但不输出唯一逐卡峰值。问题应聚焦，例如：“目前只有 EP=32，请补充 P/D 的实际 Attention TP/DP、PCP/DCP、PP 和每实例设备数；这些决定 KV 与 state 是分片还是复制。”

- 每请求有自己的历史与窗口边界；公共前缀、retained checkpoint 和 connector 持块另计，容量分摊与物理去重分别说明。
- EP 不自动切分 Attention KV。按 Attention 的 TP/CP/PP/DP 和实际 head ownership 算逐 rank。头不均匀时枚举身份；不能用 `ceil(n_kv/TP)` 推断任意映射。
- 量化 KV 不代表 FP32 compressor state 同时量化。量化 metadata、页内 padding 和反量化 workspace 分开。
- 更多驻留请求只有在消除 admission 等待、低填充或执行气泡等损失时才提高吞吐。Roofline 交点不等于算力打满，缓存预算也不证明系统 P-bound。

## 复算与审查

对已确认“各组独立消耗池 block”的几何，可用零依赖脚本：

```bash
python3 scripts/calculate_pool.py --input /path/to/layout-and-requests.json
```

输入和边界见 [日志算例与输入格式](references/worked-example.md)。不确定布局时先补证据，不使用示例默认值代替部署配置。脚本只复算预算，不运行模型或预测吞吐。

检查：组件无遗漏；跨 dtype 字节正确；压缩前后 token 单位正确；共享页不重复；请求 F 未超过全局上界；截断/边界/retention 未漏；单位一致；快照与日志是否对应。比较旧估算时逐项解释结构、dtype、block、group、MTP、async、长度和单位差异。

输出至少包含：条件与版本、组件表、公式及 bytes/MiB/GiB、P/D 与多请求作用域、日志交叉验证、尚未确认项。仅有静态证据时称“条件预算”，不称实测峰值，不将本次数字写成模型通用常数。用户只要求分析时，不修改启动配置或服务。
