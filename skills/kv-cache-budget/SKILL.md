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

## 按目标检查参数完整性

先从完整启动命令、最终解析配置、模型构造、相关环境变量和运行日志提取参数；默认值从运行版本解析，不能把“启动命令未填写”视为关闭或取1。将必要信息标为已确认、假设或待补充，仅对会改变当前结果的缺失/冲突项集中补问。

| 用户目标 | 需要确认的输入 |
| --- | --- |
| 长期有效状态量 | 模型/config版本、Attention层组成与owner共享关系、实际缓存路径和各组件dtype/scale、实际历史长度及生成增长；多模态时确认进入语言主干的实际token数及另存的encoder/cross-attention状态 |
| 逐rank运行/峰值预算 | 上述输入，以及硬件具体型号、后端/缓存kernel路径、P/D并行布局、block与page布局、Hybrid管理器、当前chunk与异步在途、MTP/DSpark、窗口回收和额外状态保留 |
| 并发上限、admission或OOM | 上述输入，以及每rank实际可用KV池/块数、请求组合与实际活跃数、前缀共享/取回、checkpoint与connector持块；判断整机OOM时另加权重、activation、workspace、通信/graph缓冲等非KV峰值 |

对逐rank预算和容量问题，按相关性逐项检查：

- **硬件和执行路径**：具体芯片/SKU、运行版本、Attention backend、量化/打包分支；相同模型在不同硬件路径的记录宽度、scale与padding可能不同。
- **Hybrid与分页**：Hybrid KV Cache Manager是否实际启用，是否转为其他管理器或全历史保留；最终生效block/page大小、缓存组、自动填充及别名规则。
- **实际负载**：输入/已计算/命中/输出的整数token数，命中长度按后端粒度对齐；多请求用各自长度和在途量。`max_model_len`、`max_num_seqs`是配置上限，不能代替实际请求长度/活跃数；预测混合负载峰值时需要长度与并发组合，均值不足以唯一确定峰值。
- **调度与推测**：chunked prefill是否启用、有效batch token预算、async和最大并发批次；MTP/DSpark的method、推测token数、实际lookahead、附加cache层/模块及放置、extra trailing retention，按代码避免重复计入lookahead。
- **前缀与保留**：本地prefix caching开关、外部池命中路径、连续前缀实际长度/共享身份/命中粒度、检查点保留间隔、淘汰和抢占/重算规则；只在确实产生额外持块或副本时计量。
- **PD交接**：connector种类、P/D角色、逐层或全量传输、D是否已有同一前缀、传输完成确认与释放时点、等待交接的完成请求数。预分配通信缓冲与KV页分开；不凭链路带宽猜测已持块峰值。
- **容量供给**：优先使用每rank实际KV池bytes、num_blocks及override后的值；确认显式KV内存预算、`gpu_memory_utilization`或同等配置、`num_gpu_blocks_override`等在该版本中的优先级。没有实测池大小时，只能按扣除已知非KV开销的可用内存做条件估算，不能将“总显存×利用率”全当KV池。

缺少核心配置时可这样提示：“当前可以计算长期有效KV；若要给逐卡峰值/并发上限，还缺P/D完整启动参数、实际Attention并行、Hybrid/异步/MTP配置和逐rank KV池日志。已知项无需重复提供。”按当前任务缩减问题，补充前提供带假设的公式或区间，不给唯一的实际峰值/并发上限。

### block_size取值约定

优先采用实际启动参数经后端校验/覆盖后的最终生效值；其次从用户指定版本的vllm-ascend及配套vLLM代码读取模型、硬件、Attention backend默认值和覆盖链。不能把通用CLI初始默认直接当作模型最终值。若仍无法确定，按用户约定的`block_size=128`做条件计算，在参数表标注“约定回退值，非部署实测/代码已确认值”；不要仅因缺少该参数反复中断询问。

block_size不是chunk_size、压缩后storage_block_size或固定ring行数。已找到代码生效值或用户明确指定其他值时，不用128覆盖；若代码显示128不受支持，说明冲突并改用可证实的候选值或补问，不强行出数。默认只输出采用值，不例行并列32/64/128；仅在比较block或解释padding时增加对照。

## 建立计算口径

- 长度使用整数 token；明确 128K 是 128000 还是 131072。GB=10^9 bytes，GiB=2^30 bytes。
- 对每类缓存列出：层/owner 数、记录结构、dtype/scale、压缩比、逻辑 block token 数、物理行数、有效页字节、填充页字节、窗口和额外 retention。
- 将历史拆为命中前缀、此前已计算 chunk、本轮新增记录。命中率只有在代表该请求连续 token 前缀时才能换算成长度；服务池平均命中率不是每请求实际命中长度。
- 本地已共享前缀按物理页去重；远端命中是否需要取回由 connector 决定。命中不自动减少当前 query 对旧历史的 Attention，也不自动减少 chunk 的在途 state。
- 从 allocator 明确各 group 的独立 block 消耗和别名关系。共享 backing 的 tensor views 不重复计算；用 pool bytes/block 计费后不再附加同一份 padding。

## 逐组件拆分 prefix cache 与 chunk

具体容量分析时，对每类 Attention 块的实际缓存组件分别给出 prefix cache 与当前 chunk 两列，不先合并长度只给总数。涉及 prefix、SWA、异步或 PD 时，读取 [分账与释放生命周期](references/prefix-chunk-lifecycle.md)，按其中步骤形成完整账目。

- 按物理 owner 枚举主KV、索引、folded/重排副本、SWA、compressor与递推state；consumer及别名不能重复计费。
- 明确命中前缀P、此前chunk历史Q、当前chunk C与所有未结算批次F。Q不得消失，也不得伪装成最初命中的P；异步时逐批说明F由哪些位置组成。
- 每个组件展示 prefix有效数据/物理页、此前chunk保留、当前chunk新增、固定请求状态及重合页；不适用应说明原因，未知保留页数不能填0。
- SWA必须分别核算“命中边界所需窗口”“prefix池全部保留checkpoint页”“活跃历史尾窗与chunk在途页”。窗口W不是prefix池容量，也不意味着整个prefix都应保存；活跃窗口admission边界余量不等于checkpoint真实页数。
- prefix与活跃页按实际物理身份取并集。分别交付活跃请求持块、包括可淘汰prefix的驻留量、新请求边际分配；不能用其中一项冒充另两项。
- 组件数据账转换为实际group/tuple成本后再汇总，不能给同一group内每个组件各乘一次完整pool成本。页内边界与压缩边界用累计差分，不能对两列分别ceil后直接相加。

## 计算调度与峰值

`max_num_batched_tokens` 是同一调度批所有请求的公共预算，不是每请求额度。分别记录实际调度量、实际未结算量和配置保守上界。

当源码支持时：

```text
F_global_max = max_concurrent_batches * max_num_batched_tokens
sum(F_request) <= F_global_max
```

并发批次数从运行版本和 PP/async 配置获取。两批在途不代表两个 forward 同时计算。不得给每条请求重复分配全局 F 上界。

先解析实际async默认分支、兼容性覆盖及最终值；未填写不等于关闭。若实际开启，正文主结论用异步场景，同步只能作为明确的对照。纯prefill单请求的F还受未完成输入位置数限制；D按实际每步喂入位置和未结算步数算，不把公共batch token上限都分给一条decode请求。

异步分账必须覆盖所有会写入新token缓存的Attention组件，包括主KV、MLA latent、压缩主记录、Index/folded副本及SWA，不能只扩大SWA而漏掉长期组件的第二批新增位置。两批等长chunk的新增记录在对齐且产出线性时约为一批的两倍；prefix、已完成历史和固定state不整体翻倍。正文分列P、已安全完成Q、各未结算C_i，再以`H=P+Q+sum(C_i)`对账。若H已经包含多批在途位置，不能再加一份chunk；比较同步/异步增量时固定已结算历史，不能混用固定H与固定已结算历史两个基准。

对采用该公式的 sliding-window manager，页数保守预算为：

```text
n = ceil(min(max_model_len, W - 1 + extra_retained_tokens + F) / block_tokens)
    + boundary_slack_blocks
```

边界余量、回收时点、MTP lookahead、prefix checkpoint retention 均从实现取证。这是 admission 上界；不能仅凭 Spec 就证明实际每行 state 都已写入，或所有组件上界同时达到。若实现采用固定 ring/circular state，改用真实固定槽规则，不能套 chunk 线性增长。

PD 分别按 P/D 实际 dtype、并行布局、在途 token 与持块规则核算。交接双端重叠量只有在寿命重叠时相加。输出 token 引入的历史增长按实际喂入模型的位置数计算；首个输出来自 prefill。

PD必须沿实际connector的`request_finished[_all_groups]`及调度器释放调用链检查`delay_free_blocks`，区分传输延长原页寿命、consumer在途写入fence的`defer_block_free`、以及可淘汰prefix保留。已确认layerwise路径返回False时，不附加“完成后等待D拉取”的请求持块，但不能推断所有connector或所有释放保护均关闭；layerwise物理buffer reuse若重写布局，须重算pool成本。

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

脚本仅验证所提供的几何、整数范围和公共在途额度，不会自动解析源码、识别设备或检查上述部署参数是否齐全，也不直接计算共享去重、交接积压或并发上限；调用前由分析流程确认输入适用范围。

检查：组件无遗漏；跨 dtype 字节正确；压缩前后 token 单位正确；共享页不重复；请求 F 未超过全局上界；截断/边界/retention 未漏；单位一致；快照与日志是否对应。比较旧估算时逐项解释结构、dtype、block、group、MTP、async、长度和单位差异。

输出至少包含：条件与版本、组件表、公式及 bytes/MiB/GiB、P/D 与多请求作用域、日志交叉验证、尚未确认项。仅有静态证据时称“条件预算”，不称实测峰值，不将本次数字写成模型通用常数。用户只要求分析时，不修改启动配置或服务。

## 详细计算过程的交付要求

容量核算输出必须让读者可以从原始参数独立复算，不能只给最终GiB、泛化公式或脚本结果。对当前问题相关的组件，按“参数来源 → 符号公式 → 数值代入 → 中间结果 → 最终汇总”展开；使用静态源码时标注仓库/commit及文件位置，日志标明rank/池/配置，假设不得伪装成已确认值。用户明确要求简短结论时可在正文给摘要，但应提供可查看的完整计算附件或文件；仅询问概念或流程、不要求具体容量时无需强行生成数值账。

1. **条件与变量表**：写明模型config、硬件/后端版本、P/D角色、rank与并行归属、长度、命中/共享状态、dtype、block/page、调度及MTP设置；每个关键值标注来源或假设。明确哪些是实际值、配置上限或保守场景。
2. **有效数据量**：分别代入主KV、索引、SWA、compressor/递推状态公式，列出层/owner数、记录数、记录有效字节。每个组件必须分列命中prefix、先前chunk、本轮新增/多批在途和固定状态；SWA另列全部prefix保留页与命中所需尾窗的区别，不能只展示合并历史总量。已共享的物理前缀不重复相加。
3. **物理页布局**：展开一条记录的数据与scale字节、每页记录行数、原始token覆盖范围、页内对齐和padding。别名view只计一份。逐项列出填充前后差额，不能只说“包含padding”。
4. **在途与页数**：从实际版本的并发批次规则推导全局F上界，并给出各请求F分配；代入窗口、extra retention、截断、向上取整及边界余量，显示每组整数页数。实际值与admission上界分表或分列，不把不能同时发生的上界称为实测峰值。
5. **缓存池与并行汇总**：若使用共享池，展示自动group/tuple分组结果、slots、各页bucket及bytes/block的数值推导或日志比值，再用独立block消耗汇总。逐rank列出组件切分、复制、层归属，不能把整模型池机械除卡数。P/D分别算，交接重叠、checkpoint/connector持块及共享去重需说明计入/排除依据。
6. **最终对账**：保留整数bytes，最后转换MiB/GiB及必要的十进制GB；各行之和应等于总计。列出有效载荷、各层物理页合计、统一池预算和实测值中可获得的不同口径，并说明差异。对日志capacity给出分母对应的最大长度/在途预算，显示复算与日志偏差。

推荐组件表（按需拆成有效载荷表与物理池表，避免混用成本单位）：

| 组件/组 | 参数与来源 | 公式及数值代入 | 记录/页/池block数 | 对应单位字节成本 | 总bytes与MiB |
| --- | --- | --- | ---: | ---: | ---: |
| 主KV、Index K、SWA、各类state等分别列行 | config/源码/日志或假设 | 展开乘法、ceil/floor与截断 | 标明计量单位 | 不混用页成本与共享池block成本 | 显示中间和最终结果 |

计算量大时将逐层/逐rank明细保存为JSON/CSV/Markdown附件，正文保留决定结果的代入过程和总计。保存文件需遵守当前环境的文件/产物规则。脚本输出可作为复算证据，但不能代替参数来源与布局适用性说明。证据不足的项明确标为待确认，并给条件公式；不填造唯一数字。
