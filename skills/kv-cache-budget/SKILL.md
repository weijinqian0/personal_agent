---
name: kv-cache-budget
description: 根据模型配置、推理后端源码与运行日志，逐组件计算单请求 KV Cache、prefix、chunk、窗口和压缩器状态占用，并核对物理页与缓存池需求。
---

# 单请求 KV Cache 核算

范围：一个请求、一个明确执行快照，按实际缓存 owner 的 shape 计算。保留模型组件与物理存储核算；不处理多请求并发、跨请求共享、P/D 部署或多卡切分，不要求 TP/DP/PP/CP 参数，也不将结果除以卡数或称为逐卡结果。

## 先看证据，再计算

默认分析 vLLM + vLLM-Ascend；用户指定或实际部署证据优先。先查可用的模型 config、后端源码、启动配置和 KV 日志，记录版本与生效路径，不能只凭模型名套公式。默认后端不代表已确认其版本、dtype 或布局。

1. **列组件**：按真实构造枚举主 K/V 或 MLA latent、压缩记录、Indexer/folded、SWA、主/索引 compressor state、递推/卷积状态，以及实际 MTP/DSpark owner。结构相同可合并，记录 owner 数、shape、dtype、scale、窗口与压缩率；别名只计一次。
2. **查状态生命周期**：每种 state 查清写入、prefix 恢复、保留、释放与去重。固定 ring/递推状态按槽位和版本数计，不随 chunk token 数盲乘；FP8 KV 不代表 FP32 state 也变成 FP8。
3. **定请求快照**：明确命中 prefix、已结算历史和每个未结算 chunk；不能用全部未命中长度替代当前 chunk。
4. **逐组件分账**：先算记录字节，再按实际页边界与物理并集去重，最后映射到 allocator；不能直接把 payload 当池需求。
5. **对账后汇总**：有日志则核对同一执行快照、同一池与同一分配范围。缺项给公式和已知小计，不虚构完整部署总数。

## 请求与 chunk

- `L`：参考序列长度；`P`：有效命中 prefix；`Q`：命中之后已安全结算的历史。
- `C_i`：各个未结算 chunk 的位置数；`F=sum(C_i)`；`S=P+Q`；`H=S+F`。
- `L-P` 是剩余参考长度，不是当前 chunk；纯 prefill 的 F 不能超过剩余未处理输入。
- 用户未给长度/命中率时，按 `L=131072`、`h=90%` 的估算假设补齐缺失项。实际命中对齐粒度 g 从后端取证：`P=g*floor(floor(L*h)/g)`。未知 g 时保留条件，不当作普通 block_size。
- 90% 命中表示复用已有 prefix，并不让历史 KV 容量直接乘 10%。缓存未开启或不支持时以实际证据说明。
- 异步开启时逐个列出 C_i；先算安全结算基线，再算含全部未结算位置的新增页。H 已含 F 时不再加一份 chunk；不默认两批在途。

压缩记录使用累计差分：prefix 为 `R(P)`，Q 为 `R(P+Q)-R(P)`，第 i 批为 `R(S+sum(C_1...C_i))-R(S+sum(C_1...C_(i-1)))`。不能分别对每个 chunk 除压缩率后取整。

## 保留的核心模块

| 模块 | 核算要点 | 需要时读取 |
|---|---|---|
| Attention / Index / 状态 | 记录 shape、dtype、scale、压缩边界、固定槽 | [组件公式](references/attention-formulas.md) |
| Prefix / chunk / 窗口 | checkpoint、活跃尾窗、在途新增页、物理并集 | [生命周期](references/prefix-chunk-lifecycle.md) |
| 源码与日志 | 实际注册、spec、分配/reshape、写入恢复与释放 | [源码追踪](references/source-tracing.md) |
| 物理布局 | 页内 padding、group/tuple、alias、真实 backing | [布局核对](references/physical-layout-validation.md) |
| MTP / DSpark | target/draft owner、验证位置、state 版本与回滚 | [投机缓存](references/speculative-cache.md) |

投机缓存保留在预算范围内；先查实际方法与模块。附加层数不等于推测 token 数，不随意补造 draft 模型或长度。确认关闭则说明；未知则列补充项，不能默认为零。

## 输出与校验

先给参数来源和组件表，最后汇总。每个组件至少列：

| 组件 / owner 数 / dtype | Prefix P | 已结算 Q | 各 chunk C_i | 窗口 / 固定 state / 额外保留 | 唯一页数 × 页字节 | bytes / MiB / GiB | 来源与状态 |
|---|---|---|---|---|---|---|---|

每格给公式、代入或明确的未知依赖；不适用需说明依据。SWA/state 分开核算命中边界所需尾窗、实际保留 checkpoint 和活跃页，不能只套一个窗口 W。长历史分项之和须等于 R(H)，页边界只计一次。

区分 `payload_bytes`、`component_page_bytes`、`request_pool_need_bytes`、`allocated_pool_bytes`、`observed_peak_bytes`。同组组件不各乘一次完整 tuple 成本；池内需求不再叠加到预分配池。未完成 allocator 映射时写“物理池需求尚未算完”；静态计算不称实测峰值。

简单几何可用 `scripts/calculate_pool.py` 复算，输入见[单请求算例](references/worked-example.md)；声明布局可用 `scripts/audit_layout.py` 核对。它们不自动证明源码真实或组件完整。

交付具体物理池需求报告时，用[结构化校验入口](references/release-gate.md)检查组件覆盖、生命周期、prefix/chunk 分账与布局成本。概念解释或用户明确只求有效量时不扩大范围。

缺失项先从现有资料查找，仅将改变结果的关键未知集中补问；可先输出完整符号账与已知小计。用户只要求分析时不修改部署配置。
