# 完整推理 HBM 占用指南

本指南把权重、KV Cache、激活与运行时三份账合成逐 rank、逐阶段的部署容量预算。适用于 vLLM/vLLM-Ascend 推理；训练还需要梯度、优化器和反向保存激活，不能沿用本总数。这里规定核算方法，不预设任何模型的固定 GiB。

## 1. 先定义要回答的容量

| 口径 | 回答的问题 | 不能替代的结果 |
| --- | --- | --- |
| Checkpoint 文件/张量字节 | 权重文件多大 | 加载、量化转换、切分后的设备常驻权重 |
| 常驻权重 | 本 rank 实际保留哪些参数与副本 | 完整 HBM |
| KV 有效状态 | 历史位置实际需要多少数据 | 对齐后的请求持块与整池分配 |
| 请求物理持块 | 活跃请求、共享前缀、在途和保留页用了多少块 | 预分配 KV 池总量 |
| KV 池分配 | 设备为缓存保留多少 backing | 当前请求有效数据量 |
| 张量存活峰值 | 同时存活的唯一物理 storage 字节 | allocator reserved 或整设备占用 |
| 设备占用峰值 | 张量、分配器额外占用、非框架运行时等占用 | 吞吐、延迟或部署稳定性保证 |

保留整数字节，最后换算：GB=10^9 bytes，GiB=2^30 bytes，MiB=2^20 bytes。128K 等长度必须写成整数 token。逐卡结果按最紧张的 rank 判断，不能按平均卡容量判断。

## 2. 一份共享配置与证据表

先从已有部署命令、配置、环境变量、源码和日志提取，三个专项复用同一版本和负载。每项标注“已确认 / 条件假设 / 待确认”和出处；默认值要追踪覆盖链。源码记录 commit、相关工作区修改和文件位置；日志记录 rank、角色、阶段和采集条件。

| 输入 | 核对内容 |
| --- | --- |
| 模型 | revision、config、checkpoint index/header、实际加载的视觉/MTP/draft/其他模块、权重共享关系 |
| 环境 | 芯片/SKU、每卡容量及当前可用量、vLLM/vLLM-Ascend、TorchNPU/op-plugin、CANN 及实际 backend |
| 放置 | P/D 或混部、实例/节点/rank 映射、PP 层分布、Attention TP/DP/PCP/DCP、有效 EP、专家内部 TP、共享专家 DP、其他独立进程组 |
| 数据格式 | 权重/激活/通信/KV/state 各自 dtype、量化方式、scale、padding、NZ/打包格式 |
| 负载 | 各请求输入/已计算/生成长度、实际活跃数、共享前缀身份、每 DP 实际调度 token、批 token 上限与异步在途 |
| 缓存 | 最终 block/page/group、Hybrid、窗口与 checkpoint、connector/延迟释放、KV 池显式预算或实际分配策略 |
| 运行分支 | 投机方法、draft/target 配置和长度、图模式/档位、MoE dispatch 分支、通信池、EPLB、offload 与设备缓存 |

EP 不能唯一确定 TP/DP，也不自动切分 Attention KV。上限参数不能当实际负载，模型 config 中存在 nextn 层不能证明投机已开启。缺少决定结果的配置时集中补问，同时继续独立分项。用户也无法确认时保留符号化未知项及取证方法。

## 3. 三个 skill 如何交接

### 权重账：weight-hbm-estimation

以 config、checkpoint header 与实际加载路径核对：embedding/head、Attention/Indexer/compressor、Dense FFN、router、路由/共享专家、Engram、norm 和实际附加模块。逐 rank 追踪切分轴、复制、量化元数据、转换后 dtype/format、EPLB 冗余、offload 与额外权重副本。

基本式为本卡唯一权重 storage 字节之和；`numel*dtype_bytes` 仅适用于相应稠密物理格式，打包、scale 和 padding 另核对。MoE 按本卡驻留专家数，不能按 top-k 激活比例缩小。offload 只扣真正卸载的部分。转换期间旧新权重共存记在加载阶段。

交接：逐 rank 常驻权重 W、转换/临时权重副本及其生命周期。该 skill 附表中出现的非权重共享缓冲移交运行时账，不能两边都加。

### 缓存账：kv-cache-budget

按物理 owner 列主 K/V 或 MLA latent、Indexer/folded、SWA、compressor/递推状态、投机 draft/target 附加缓存。先求有效数据，再转实际页/group/tuple 成本。示例结构式 `L*H*2*n_kv*d*b` 仅用于等宽等 dtype 的 K/V；紧凑 MLA 不机械乘 2，混合结构逐组件核算。

按该 skill 保留 prefix、已安全结算历史 Q、各未结算批次和固定状态；共享前缀按物理页去重，公共 batch token 上限不逐请求复制。压缩和页边界用累计差分，不能对每个 chunk 独立取整后相加。SWA、固定 ring、prefix checkpoint 各按实际生命周期算。

默认分析多请求持续供给，以 N/请求组合表达未知并发；单请求为辅助。默认预算范围包含投机，但是否启用、方法和长度仍需取证。`block_size=128` 只是无法确认时的约定条件回退，不覆盖有效配置或不受支持路径。

交接两份结果：请求需求 K_need（按缓存组给整数块数）和设备实际池 K_pool。请求需求已经包含于整池时，总 HBM 只加入 K_pool；池外独立 state 仍单列，不能因名称含 KV 就遗漏。

### 激活及运行时账：activation-hbm-estimation

沿实际执行路径记录 hidden/residual、Attention/Indexer、Dense/MoE、AllGather/AllToAllV、logits/sampler、投机/多模态、workspace、HCCL/MC2/SHMEM、图池、持久缓冲、非框架运行时和分配器额外占用。

Prefill 主要使用本轮实际 chunk/query token，decode 与投机使用真实执行位置；不是按最大历史长度乘所有激活。有效行数不等于实际分配 shape，图档位、capacity、padding 和多流会改变峰值。普通串行中间激活不乘所有层，PP/异步/多流共存另建快照。

交接：非权重非 KV 的逐阶段唯一 storage、持久项和未知项。权重聚合/转换副本归权重账；临时 KV gather/打包/反量化 staging 归本账；持久 cache/state 按 KV 归属交接。

## 4. 用物理归属消除三份账的重叠

统一账本建议至少保留：

```text
storage_id, role, rank, owner_skill, category, shape, dtype, physical_bytes,
aliases, backing_pool, live_phases, status, evidence
```

`physical_bytes` 未知用 null 并附符号。静态分析可用可追踪的 backing/owner key；不要编造实际设备地址。同一身份只保留一项，别名的生命周期取并集。

| 容易重叠的项 | 合账方法 |
| --- | --- |
| 共享参数、view、transpose | 追踪同一 backing；独立 clone/contiguous 才可能新增 |
| KV 池与请求页 | 整池用于设备总账；页需求只做池内供需对账 |
| compressor/state | 持久缓存归 KV；权重归权重；临时投影/workspace 归激活 |
| 权重格式转换、AllGather 权重副本 | 权重账含生命周期；激活账排除同一副本 |
| 图池与捕获张量 | 用池整体或内部张量加池剩余，不能两份全额相加 |
| HCCL 与 non_torch | non_torch 已包含该通信分配时扣重；不能再加同一域 |
| allocated 与 reserved | reserved 已含 allocated；只另加有依据的 slack，不能相加两个总量 |
| 安全余量 | 独立决策项，不冒充已测量运行时或给未知项填固定百分比 |

三者提供的 Python 工具只验证输入范围内的算术，不自动发现漏项或推断 alias。KV 的 `calculate_pool.py` 用于已确认独立 group 消耗的请求预算；激活的 `calculate_peak.py` 排除权重和 KV，不能把它的结果称为模型总 HBM。

## 5. 先按时间合账，再取最大值

令 S_r(t) 为本 rank 在时刻 t 存活、已按物理身份去重的 storage 集合：

```text
M_r(t) = sum(physical_bytes(s) for s in S_r(t)) + X_r(t)
M_peak_r = max_t M_r(t)
M_peak = max_r M_peak_r
```

X 是 storage 清单之外、尚未计入的分配器额外占用和设备运行时项；已在池/设备差分中覆盖的部分不再加。所有部分必须同口径。只有确认 W 与 K_pool 在相关阶段常驻且其余项互不重叠时，运行阶段才可简写：

```text
M_run_r = W_r + K_pool_r + max_t(A_r(t) + P_r(t) + X_r(t))
```

A 为临时激活/workspace，P 为去重后的其他持久池。各专项已经包含的持久项不再追加。加载阶段可能尚无 KV 池、图池；后续图池也可能与 eager prefill 共存，按真实初始化顺序确定。

至少覆盖相关阶段：加载/转换、通信初始化、warmup/profile、图捕获、prefill、decode/投机、混合调度及 PD 交接。加载峰值和运行峰值分别报告；profile/capture 临时分配也可能决定能否启动。异步 collective、跨流引用和缓存分配器会延长存储占用，Python 局部变量消失不证明设备已释放。

例如某个纯教学场景中，常驻权重 20 GiB，prefill 激活 8 GiB，此时尚无 KV 池；decode 时 KV 池 10 GiB、激活 2 GiB，且本例无其他开销。总峰值是 `max(20+8,20+10+2)=32 GiB`。把专项各自峰值相加得到 38 GiB 只构造了未同时发生的组合，不能称实际峰值。本例不用于真实部署默认值。

没有证明全部阶段上界可同时达到时，独立上界之和仅标“保守预算界”；缺少任一项的上界，不能称保证上界。未知 U/G 可能改变峰值阶段，保留逐阶段表达式，不只给已知项最大值加一个含糊修正。

## 6. KV 供需与能否装下

1. 先求同一配置下非 KV 峰值/分配顺序，再追踪实际版本的 KV 自动预算、显式预算和 override 优先级。`HBM容量*利用率` 不能直接当 KV 池。
2. 自动预算与 profile 负载相互影响时，按真实后端分配流程核对，不先假定一个 KV 池反推出自己。无日志时给条件供给式，不伪造实际块数。
3. 对每个 rank 和独立缓存组检查 `need_blocks <= available_blocks`。非互换缓存组不能只合成总 bytes 判定；prefix 可淘汰页、不可释放在途/交接页分别核对。
4. 对每卡计算 `headroom_r = usable_capacity_r - M_peak_r - explicit_margin_r`，各卡都应满足；明确 usable_capacity 是否已扣其他进程，避免再次扣除。PD 分别检查 P 与 D，节点总量只作设备总和，不能代替单卡适配。
5. 未知 N 用请求组合与 N 的条件式。仅在等长、无共享、相同放置和固定额外保留等已证实简化下可用剩余块除单请求成本；一般要按各组整数需求与请求组合求 N。

“静态预算可装下”不证明带宽、吞吐或时延达标。设备占用未知、峰值阶段未覆盖或物理 padding 未确认时，给已知小计、条件余量和缺项，不给无条件可部署结论。

## 7. 交付可独立复算的报告

建议报告顺序：

1. 结论与适用配置：是条件预算、源码已确认容量还是实测，列关键未知。
2. 共享参数/证据表及三个专项的调用与覆盖范围。
3. 权重、KV 有效/物理/整池、非权重非 KV 分账，展示“参数来源 → 公式 → 数值代入 → 中间结果 → 整数 bytes”。
4. 同 rank 同阶段合账：`角色/rank | 阶段 | W | K_pool/池外state | 临时项 | 其他持久项 | 分配器/运行时 | 总 bytes/未知式 | 证据`。
5. 最大单卡、最小余量、P/D 各自结果、KV 各组供需与并发条件；说明加载与稳态峰值差异。
6. 去重说明、未覆盖项及获取方法。大明细可存 JSON/CSV/Markdown，正文保留关键代入和汇总。

若用户已授权且有设备条件，可固定相同配置在上述阶段同步采集 allocated/reserved、device free、缓存几何与峰值日志。free 差分可能混入其他进程或同期初始化，不能直接作为纯激活。没有实测条件时交付静态条件预算，无需下载完整权重、执行远程模型代码、改服务配置或启动昂贵 profiling。

## 8. 下次调用方式

在本仓库直接说：

> 估算模型 X 在给定部署命令下的完整 HBM 占用，结合权重、KV Cache、激活和运行时三个 skill，给逐卡分账、阶段峰值与剩余 KV 容量。

安装到 Agent 的 skills 目录后也可显式调用：

> 使用 $model-hbm-budget 估算模型 X 的 HBM，占用按实际部署配置核对；缺少关键值先完成可独立部分，再集中补问。

如果只需要权重、KV 或激活，则直接调用对应专项 skill。总入口负责路由与合账，不强制并行 Agent，也不把一次请求解释成对外发布或设备运行授权。
