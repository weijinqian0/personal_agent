---
name: kv-cache-budget
description: 根据模型结构、推理后端源码和运行日志估算 KV Cache、窗口与压缩器状态容量，分析 chunk、前缀命中、异步在途、PD 分离及多卡部署的缓存预算差异。适用于 vLLM/vllm-ascend 等推理引擎的容量核算与日志对账。
---

# KV Cache 容量与峰值分析

交付可复算的逐组件、逐请求、逐 rank 缓存账。区分三个结果：长期有效状态字节、请求消耗的物理池块预算、实测持块峰值。另行说明预分配池总量；它不等于请求当前使用量，也不等于整机显存峰值。

## 默认推理后端背景

用户未另行指定、且实际部署证据未表明其他后端时，HBM、权重、KV Cache、激活及运行时占用分析默认采用 **vLLM + vLLM-Ascend（Ascend 执行路径）**，不再重复询问后端名称。用户显式指定的其他后端或已确认的实际部署优先；发生冲突时说明差异，不混用执行路径。

默认后端只确定分析路径，不确定硬件代际、vLLM/vLLM-Ascend 版本、KV dtype、并行布局、chunk、异步、MTP或缓存池配置；这些仍按实际代码、配置和日志取证，关键未知项保留条件并集中补问。优先追踪配套vLLM与vLLM-Ascend的注册、替换及最终生效路径；只找到上游vLLM实现不代表Ascend插件沿用同样布局。模型官方参考实现仅用于解释结构或明确标注的辅助对照，不能替代默认后端的物理容量账，也不能因缺参数而静默切换到参考实现或BF16布局。

## 容量计算交付门禁：先逐模块分账，再汇总

### 受控报告入口

具体部署需求/逐rank容量报告按[交付门禁](references/release-gate.md)建立结构化证据账，并运行 `scripts/release_budget.py --input ledger.json --review review.json --output-dir <目录>`。入口核验源文件指纹和版本绑定、完整组件inventory与状态生命周期、P/Q/各批chunk/窗口/额外保留分账、公共在途额度、物理映射及三路成本，并要求与当前账本绑定的独立复核。只在退出0时交付生成的部署/条件报告；退出非零时交付生成的缺口与已知小计，不在聊天中手写一个绕过门禁的总数。只询问概念或明确只求静态有效量时沿用原范围。

先填[账本模板](references/ledger-template.json)，未知保留unknown；读过skill、算术正确、旧的 `audit_layout.py` 通过或生成过文件，都不等于本入口通过。每个state必须提供write/restore/retain/release/deduplicate来源；不能只根据shape补表。独立复核不是计算者自行改名签署；更改输入或证据后重新复核。门禁只能约束使用该入口的产物，不能技术上拦截任意自由文本，CI也不能自动判断源码解释真实；不得宣称已保证所有Agent回答。

### 部署总量必须完成物理布局映射

先区分五种口径：有效载荷 `payload_bytes`、组件分页小计 `component_page_bytes`、请求物理池需求 `request_pool_need_bytes`、预分配池 `allocated_pool_bytes`、实测峰值 `observed_peak_bytes`。未知字段保留待确认/null，不互相替代；admission上界标为上界，不称实测写入量。

对部署容量/逐卡预算，组件页之后必须追到实际生效的allocator：确认每个owner的group、backing/alias、分配单位、tuple/bucket、空槽及跨组件padding、独立block消耗和rank归属。按独立页、共享池、多池或固定ring的真实布局计费；不能强制所有模型套统一U，也不能同组各组件重复乘完整U。模型config和Spec页大小不足以证明池块成本。

**未完成物理映射时，结论明确写“部署池需求尚未算完”**，仅交付明确命名的组件小计和未知项公式。不得以“条件逐卡预算”“另加少量padding”等标签将组件小计作为部署总量。用户明确只要有效量时按其范围交付。

分配器、admission与capacity分别追踪并检查成本一致性，包含draft/MTP尾槽；有冲突时不能宣称对账完成。无日志但完整追通源码时可给“源码推导、未实测的条件池需求”；有日志时仅用同rank、同池、同分配范围的数据核对，字节/块数比值不能跨范围套用。

优先完成[物理布局核对](references/physical-layout-validation.md)的证据表，再进行部署容量汇总。已有布局仅在相关模型/后端代码、硬件、dtype、block、并行、Hybrid、draft及connector布局身份相同时复用；只变长度/chunk且未改变布局时重算需求即可。可用 `scripts/audit_layout.py` 校验声明布局的算术和冲突；脚本不证明证据真实性或实际生效路径。

本节适用于具体容量计算，无论单独调用还是由总预算skill编排。仅解释概念时不强制生成容量表；用户明确只要静态有效量时可按限定范围计算。否则，不得自行用“全部输入已处理完、F=0”的快照替代prefix/chunk执行预算。静态小计只能作为辅助对照，不能代替以下分账。

1. **先形成组件清单**：从配置与实际构造列出每类Attention及其全部物理owner，包括主KV/latent、Indexer、folded副本、SWA、主/索引compressor state、递推状态和实际投机模块。按结构相同的层合并，但层数、owner与差异必须有依据；发现的每个组件都必须出现在分账表中。
2. **每个组件分别填写prefix和chunk**：定义P为命中前缀，Q为命中后已安全结算历史，C_i为每批未结算位置，F=sum(C_i)，S=P+Q，H=S+F。L−P是剩余参考长度，不能直接标成“当前chunk”。逐批拆分C_i，不能只列合并H或F。
3. **缺参数不删列、不改场景**：chunk/批次数/释放规则未知时，仍为每个组件写出依赖P、Q、C_i的公式、已知数值、未知参数及其影响，集中补问；不默认F=0、Q=L−P或任意8K chunk。不支持prefix复用的项写“不适用＋证据”；未知写“待确认＋公式/依赖”，均不填0。不要把未知部署路径静默换成官方参考实现或BF16布局；参考实现只能作为明确标注、保留同样分账的条件场景。
4. **汇总前核对覆盖和守恒**：组件清单与分账行逐项对应；所有适用行都有prefix、Q、各C_i、窗口/固定state、共享/重合处理。长期记录的分项之和须等于R(H)，页按物理并集去重；固定state不能随C_i重复分配。窗口、checkpoint与回收按各自规则核对，不套长期记录守恒。
5. **不通过时先补表再交付**：缺行、删列、用完整L掩盖chunk、用未命中总量冒充单chunk，均视为不通过；修复后再汇总。证据仍不足时可以交付完整的符号分账和已知小计，但必须把未完成项列在结论中，不声称预算已完成。给文件附件或标“条件预算”不能豁免逐模块分账；正文至少展示组件覆盖与prefix/chunk分列，不能只把必需分账藏在附件。

必填分账表（可按组件拆表、按批扩列；有效数据与物理页成本分别使用本表）：

| Attention类型 / 层数 / 物理owner / 证据 | 组件 | 命中prefix P：记录/页、公式、bytes | 已结算chunk历史Q：差分、bytes | 各未结算chunk C_i：位置、差分、bytes | 历史窗口 / 固定state / 额外checkpoint | 物理重合、释放与去重 | MiB/GiB、rank/阶段、状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 每种实际Attention结构逐组件列行 | 主KV、Index、SWA、各state等 | 数值或符号公式；不适用需证据 | 不得并入P或漏掉 | 不得用L−P代替单批C_i | 与prefix/活跃页取并集；固定槽单列 | alias、CoW、padding、延迟页按实际实现 | 已确认 / 条件计算 / 待确认 |

最终答案发出前按本表逐行检查；数值正确但缺少prefix/chunk组件分账，仍不满足交付要求。

## 默认负载与输出顺序

### 默认序列长度与命中率

用户未特别指定且实际负载没有确定取值时，序列参考长度L默认128K=131072 tokens，命中率h默认90%，表示每请求连续prefix的估算假设。用户或已确认部署/负载明确的字段优先，只对缺失字段回退，不因缺少长度/命中率重复提问。默认L不是max_model_len，也不是每轮chunk长度；输入/输出拆分和后续decode增长没有证据时单独保留条件，不补造数值。

有效prefix按运行后端命中对齐粒度g计算：`P_target=floor(L×h)`，`P_effective=g×floor(P_target/g)`，`未命中参考长度=L-P_effective`。g来自实际后端的prefix粒度与缓存组对齐约束，不能无条件当普通block_size。多组件压缩/分页继续按累计位置差分，未命中参考长度不等于单轮调度token数。

条件算例g=128：L=131072、h=90%、P_target=117964、P_effective=117888、未命中长度=13184 tokens；有效命中率117888/131072=89.94140625%。用表格展示目标与对齐后命中，不宣称实际精确命中90%。

90%假设不证明prefix caching实际开启、不同请求物理共享、远端命中无需回传或只保存10%历史。实际关闭prefix时说明实际命中为0的依据；90%仅保留为有条件的假设场景，不修改服务配置。已确认服务平均命中率不能当每请求连续prefix。命中prefix的实际保留、取回和去重仍按物理身份核算，K_pool及权重不会因90%命中自动乘10%。

按用户约定，未明确限定单请求时，默认分析多请求持续供给，以多请求每rank总账为主结论；单请求只作单位成本辅助。多请求不等于已知并发数或必然填满batch：缺少实际持块数时用N表示，并给明确请求数/阶段的条件示例，不擅自固定为2/3条。区分排队、已接入持块、在途计算与D侧请求，分别确认共享前缀；90%命中不意味着请求之间共享同一物理prefix。

最终账单按“去重后的命中prefix → 已结算历史Q → 活跃历史窗口/固定state → 异步在途增量 → 额外保留”汇总。逐组件prefix/chunk明细仍保留；先形成截止安全结算位置S=P+Q的基线，再在最后统一加所有未结算批次（含当前批）的新增页。详细差分与公共额度规则见[分账与释放生命周期](references/prefix-chunk-lifecycle.md)。这只是展示和归因方式，不改变异步实际生命周期，也不允许在已包含F的完整历史H上重复加chunk。

按用户约定，默认容量预算包含投机解码相关缓存，不再静默采用“无MTP/无draft”。这是估算范围约定，不是宣称后端默认开启投机，也不授权修改启动配置。先从实际配置/代码确定方法、draft模型/原生MTP模块和推测长度；缺失时按[投机缓存核算](references/speculative-cache.md)给出有依据的含投机场景或待确认项。只有用户明确排除或核实实际部署关闭时，才以不含投机为实际配置结论，并注明范围差异。

## 获取证据

1. 查找用户指定的本地仓库和部署日志，记录模型 config、后端 commit、未提交相关修改及实际启动参数。有多个 checkout 时明确使用哪一个。缺少资料仍可给条件公式，但不制造精确部署数值。
2. 根据模型 config 和实际构造路径列出缓存组件：全局 K/V 或 latent、索引、SWA、主/索引 compressor state、递推/卷积状态、MTP/DSpark。需要结构公式或不同Attention对比时读取 [Attention 与状态公式](references/attention-formulas.md)，使用其中prefix/chunk双列公式表。
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
- **投机预算完整性**：默认计入draft/MTP独立KV、主模型验证lookahead、状态版本/回滚槽及持久hidden-state cache；逐P/D确认实际加载/写入与共享owner，不能只增加主模型几枚token而遗漏draft长历史。模型config有nextn层不证明运行启用，也不能将nextn层数当推测token数；缺少方法或draft配置时不把未知缓存记0。
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

用户要求梳理不同Attention公式时，用一张表按“MHA/GQA/MQA、紧凑或展开MLA、压缩主KV、Indexer/folded、SWA、各类state”分行，分别列prefix与chunk公式；区分Attention结构和其缓存组件，一个模型可能同时包含多行。具体模型只保留相关行，并补实际owner、dtype/scale、来源和数值。表内组件数据与统一pool成本不能混为一列，Q历史、活跃尾窗及固定状态不可因双列展示而漏掉。

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

异步增量必须单独列账：`角色/rank | 缓存组件/组 | 安全结算基线块数 | 含全部未结算批次及延迟释放后的块数 | 差额块数 | 单位块bytes | 异步增量bytes | 已计入的池/类别 | 状态与证据`。先计算每组差额，再对物理owner去重；没有基线、有效B或释放规则时，状态写“尚未完成定量核算”，不能仅放入未解释的O/U后声明“已计入异步”。明确区分“需求增量已计算”和“实际峰值已验证”。

异步持块已在固定K_pool内时，异步需求增量消耗池内空闲块，不再把差额bytes加到设备K_pool；池外state、动态扩容或异步非KV持久量仅在确实新增物理storage时加入完整HBM。比较同步/异步增量须固定安全结算基线和请求组合。B批共享的公共T额度只分配一次，不能逐请求或逐组件各占一份完整额度。

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

具体容量答案最后必须给简短结论：采用的模型/执行条件、多请求每rank的P/D总账及N/阶段（未确定N时给条件公式）、单请求辅助成本、最后加入的异步在途增量、额外prefix或延迟页的补充项；明确投机方法/长度、draft与target附加缓存是否已计入，未知时用补充项而非零，纯主模型小计不得冒充默认完整预算。用户明确限定单请求时按其范围输出。不把活跃预算叫作节点总显存；未知保留量用条件式表示而非填0。用户质疑结果偏小时，从长期记录、各类在途state、padding和真实backing分配对账，不无依据把整个缓存再乘2，也不把单请求静态预算当作部署显存建议。

## 详细计算过程的交付要求

单独调用时先按模块展开参数、公式、代入和异步增量，最后给KV汇总表，区分有效数据量、请求物理持块需求 K_need、实际预分配池 K_pool、池外独立state和未知补充项；末尾汇总重复模型配置、每rank/P-D作用域、阶段和变量定义，不使用上下文指代。未知池容量保留符号，不用K_need冒充设备已分配占用。

由 `model-hbm-budget` 编排时交接组件几何、K_need、K_pool、池外state及状态；总HBM只加入实际pool与池外state，不再加池内需求。完整报告采用总入口的先详算、末尾汇总顺序，不另起一份独立报告。

每个缓存组件与中间计算步骤必须用表格输出：长度/命中对齐、prefix/Q/F分账、有效记录、物理页与padding、group/tuple、异步差额和最终池供需分别呈现。表格包含来源、符号公式、实际代入、记录/页/块数、整数bytes、MiB/GiB、rank/阶段及状态；公式代码块和文字只能补充，不能替代计算结果表。

容量核算输出必须让读者可以从原始参数独立复算，不能只给最终GiB、泛化公式或脚本结果。对当前问题相关的组件，按“参数来源 → 符号公式 → 数值代入 → 中间结果 → 最终汇总”展开；使用静态源码时标注仓库/commit及文件位置，日志标明rank/池/配置，假设不得伪装成已确认值。用户明确要求简短结论时可在正文给摘要，但应提供可查看的完整计算附件或文件；仅询问概念或流程、不要求具体容量时无需强行生成数值账。

1. **条件与变量表**：写明模型config、硬件/后端版本、P/D角色、rank与并行归属、长度、命中/共享状态、dtype、block/page、调度及MTP设置；每个关键值标注来源或假设。明确哪些是实际值、配置上限或保守场景。
2. **有效数据量**：分别代入主KV、索引、SWA、compressor/递推状态公式，列出层/owner数、记录数、记录有效字节。每个组件必须分列命中prefix、先前chunk、本轮新增/多批在途和固定状态；SWA另列全部prefix保留页与命中所需尾窗的区别，不能只展示合并历史总量。已共享的物理前缀不重复相加。
3. **物理页布局**：展开一条记录的数据与scale字节、每页记录行数、原始token覆盖范围、页内对齐和padding。别名view只计一份。逐项列出填充前后差额，不能只说“包含padding”。
4. **在途与页数**：从实际版本的并发批次规则推导全局F上界，并给出各请求F分配；代入窗口、extra retention、截断、向上取整及边界余量，显示每组整数页数。实际值与admission上界分表或分列，不把不能同时发生的上界称为实测峰值。
5. **缓存池与并行汇总**：若使用共享池，展示自动group/tuple分组结果、slots、各页bucket及bytes/block的数值推导或日志比值，再用独立block消耗汇总。逐rank列出组件切分、复制、层归属，不能把整模型池机械除卡数。P/D分别算，交接重叠、checkpoint/connector持块及共享去重需说明计入/排除依据。
6. **最终对账**：保留整数bytes，最后转换MiB/GiB及必要的十进制GB；各行之和应等于总计。列出有效载荷、各层物理页合计、统一池预算和实测值中可获得的不同口径，并说明差异。对日志capacity给出分母对应的最大长度/在途预算，显示复算与日志偏差。

使用开头的必填分账表；需要展示页几何或group/tuple单位成本时另加表，不用仅含总bytes的组件表替代prefix/Q/各批chunk分账。

计算量大时将逐层/逐rank明细保存为JSON/CSV/Markdown附件，正文保留决定结果的代入过程和总计。保存文件需遵守当前环境的文件/产物规则。脚本输出可作为复算证据，但不能代替参数来源与布局适用性说明。证据不足的项明确标为待确认，并给条件公式；不填造唯一数字。
