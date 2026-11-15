---
name: kv-cache-budget
description: 根据模型配置、推理后端源码与运行日志，逐组件计算单请求 KV Cache、prefix、chunk、窗口和压缩器状态占用，并核对物理页与缓存池需求。
---

# 单请求 KV Cache 核算

范围：一个请求、一个明确执行快照，按实际缓存 owner 的 shape 计算。保留模型组件与物理存储核算；不处理多请求并发、跨请求共享、P/D 部署或多卡切分，不要求 TP/DP/PP/CP 参数，也不将结果除以卡数或称为逐卡结果。

**物理总量交付门禁**：先固定场景与源码版本，再执行实际后端分组算法并保存输入及输出；按最终 group/bucket 复算槽数、空槽/空 bucket、页填充及唯一 backing，不凭层数手填。共享 tuple 的声明布局须通过 `scripts/audit_layout.py` 校验（不替代源码重放）。参数或源码变化后整账重算。物理映射与生命周期未闭合时，`request_pool_need_bytes=null`；不得在用户追问总数时把小计或新补的假设改称最终结果。

## Hybrid / padding 必算项

普通“KV占用/容量”请求应包含物理布局，不自行缩成有效槽位小计。遇到全注意力与GDN/Mamba、SWA、不同压缩率、Indexer或draft混合，先读[Hybrid与padding核算](references/hybrid-padding.md)，分别计算记录/尾页、原生页对齐、group/bucket空槽和MTP尾槽；不默认它们很小或为零。

状态份数/admission规则不是物理成本：`GQA有效KV + N份原始GDN状态`仍是组件账，必须继续追最终block、padded page、分组和唯一backing。量化、MTP或状态模式改变后重放布局，不直接沿用旧padding系数。缺布局证据时，主结论写“物理池需求尚未算完”，`request_pool_need_bytes=null`，不把条件小计当部署总数。

## 固定核算顺序：证据 → 快照 → 组件 → 生命周期 → 分账 → 分页 → hybrid → 对账

默认分析 vLLM + vLLM-Ascend；用户指定或实际部署证据优先。先查可用的模型 config、后端源码、启动配置和 KV 日志，记录版本与生效路径，不能只凭模型名套公式。默认后端不代表已确认其版本、dtype 或布局。

具体容量请求按以下八步推进，每步保留产物和未完成项；详细检查及输出表见[固定核算流程](references/fixed-workflow.md)。只求概念或明确限定静态有效量时按用户范围计算。

1. **锁定身份与口径**：记录模型revision、配套后端commit/相关dirty diff、硬件、生效patch和配置；区分payload、组件页、请求池需求、预分配池、实测峰值。旧算例仅在布局身份匹配时复用。
2. **定义请求快照**：列L、目标命中率、实际对齐g、P、Q、逐批C_i、F和H；区分batch公共token上限与本请求实际调度量。未知chunk不默认F=0，不把输入结算后快照替代prefill预算，也不默认两批在途。
3. **枚举全部owner**：遍历每种Attention的主K/V或latent、压缩记录、Indexer/folded、SWA、主/索引compressor state、递推/卷积及实际MTP/DSpark；登记数目、shape、dtype/scale、压缩率、窗口、alias，未知投机项保留。
4. **追状态生命周期**：逐owner验证写入、prefix恢复、保留、释放与去重。区分固定ring和滑窗分页state；FP8 KV不代表FP32 state量化。恢复checkpoint、活跃窗口和额外保留按物理并集核算，不能仅凭W断定全部prefix占用。
5. **逐组件prefix/chunk分账**：每个owner分别列P、Q、各C_i、窗口/固定槽与额外保留，给累计记录差分和整数bytes。不适用需证据，未知不填0。检查长历史分项守恒；chunk未结算期间检查全部会写入的组件，不只扩大主KV或SWA。
6. **转换唯一组件页**：记录逻辑/存储block、页内padding、边界取整及alias；chunk页按累计页集合差分归因，不能各列独立ceil后相加。标明实际持块或admission上界。
7. **映射实际hybrid分配器**：从最终owner集合重建group拆分、page-size buckets、tuple槽、MTP尾槽、空bucket/空层槽及stride/backing。正文展示组件页到物理成本的桥接。同组main/index不能各乘完整U；别名backing只算一次。不能止于Spec或用“另加padding”代替映射。
8. **三路对账后交付**：分别核对allocator、admission、capacity的单位成本与scope，有日志再核对同范围唯一backing/块数。MTP槽数等出现冲突即保留未完成，不能择一当最终需求。具体物理池报告通过[结构化交付门禁](references/release-gate.md)后发布；缺证据时仅给逐组件符号账、已知小计与缺口。

## 请求与 chunk

- `L`：参考序列长度；`P`：有效命中 prefix；`Q`：命中之后已安全结算的历史。用户写K/k时显式声明采用1000还是1024 tokens；bytes、MiB/GiB与十进制GB分别标单位。
- `C_i`：各个未结算 chunk 的位置数；`F=sum(C_i)`；`S=P+Q`；`H=S+F`。
- `L-P` 是剩余参考长度，不是当前 chunk；纯 prefill 有 `Q+F<=L-P`。`max_num_batched_tokens`是batch公共上限，不是本请求必然的C_i；可以给单请求占满上限的条件分支，但不得当成已确认实况。
- 用户未给长度/命中率时，按 `L=131072`、`h=90%` 的估算假设补齐缺失项。实际命中对齐粒度 g 从后端取证：`P=g*floor(floor(L*h)/g)`。未知 g 时保留条件，不当作普通 block_size。
- h用于连续prefix公式前先确认其语义；服务平均命中率不能直接当成本请求连续prefix。90% 命中表示复用已有 prefix，并不让历史 KV 容量直接乘 10%。缓存未开启或不支持时以实际证据说明。
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

必需的逐owner prefix/chunk表必须在答复正文展示，不能只放附件或用总数替代。每格给公式、代入或明确的未知依赖；不适用需说明依据。SWA/state 分开核算命中边界所需尾窗、实际保留 checkpoint 和活跃页，不能只套一个窗口 W。长历史分项之和须等于 R(H)，页边界只计一次。

区分 `payload_bytes`、`component_page_bytes`、`request_pool_need_bytes`、`allocated_pool_bytes`、`observed_peak_bytes`。同组组件不各乘一次完整 tuple 成本；池内需求不再叠加到预分配池。未完成 allocator 映射时写“物理池需求尚未算完”；静态计算不称实测峰值。

物理池报告另列group/bucket/slot成本表及同一快照的`组件页 → hybrid额外成本 → 已知页物理映射`桥接表。比较前后结果时先统一dtype、block、draft和P/Q/C_i；不能把配置变化或不同快照的差额全称为漏项/padding。未闭合保留项时，条件映射仍只标已知小计，`request_pool_need_bytes=null`。

简单几何可用 `scripts/calculate_pool.py` 复算，输入见[单请求算例](references/worked-example.md)；声明布局可用 `scripts/audit_layout.py` 核对。它们不自动证明源码真实或组件完整。

交付具体物理池需求报告时，用[结构化校验入口](references/release-gate.md)检查组件覆盖、生命周期、prefix/chunk 分账与布局成本。概念解释或用户明确只求有效量时不扩大范围。

缺失项先从现有资料查找，仅将改变结果的关键未知集中补问；可先输出完整符号账与已知小计。用户只要求分析时不修改部署配置。
