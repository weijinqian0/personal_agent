# Prefix、chunk与释放生命周期分账

用于具体容量分析，不是要求任意模型采用统一缓存实现。先从模型config、构造/spec、allocator、scheduler和connector取证，再按下列顺序计算；保留来源、公式、代入、整数bytes及单位换算。

## 1. 定义输入与区间

- P：真实命中且对齐后的连续前缀长度；全服务平均命中率不能直接代入。
- Q：命中后、此前chunk已经计算的历史。
- C：当前新增chunk；多个未结算批次分别列出位置区间，不重复计数。
- F：该请求已调度但未安全结算的位置数，不一定等于C，也不一定达到配置上限。
- W、extra retention、block tokens、compression ratio、dtype/scale、实际owner、缓存可否prefix复用均逐组件确认。

先标明本地命中还是外部池恢复；外部命中可能还需分配本地页，checkpoint可能在不同存储介质。按rank与介质分账。

## 2. 枚举每类Attention块和存储owner

从config到构造路径列出各层的Attention组合及共享图，再列实际持久缓存：主K/V或MLA latent、Index K、folded副本、SWA、主/索引compressor、递推/卷积state、MTP/DSpark。每一类均进入下表；未知不能略过，未启用/不可prefix复用需注明依据。

| 组件/owner | prefix有效记录与页 | 先前chunk保留Q | 当前chunk新增/多批在途 | 固定请求状态 | 物理重合与来源 |
| --- | --- | --- | --- | --- | --- |
| 每种组件一行 | 区分命中所需与整个池保留 | 不归入原始P | 列出C与F的区别 | 不随chunk重复分配 | 实际shape/spec/retention证据 |

## 3. 先算有效记录，再转换物理页

对保留完整历史的组件，令R(x)为源码在前x个原始位置产出的累计缓存记录数：

```text
prefix_records = R(P)
previous_records = R(P+Q) - R(P)
chunk_records = R(P+Q+C) - R(P+Q)
bytes = records * actual_record_bytes * actual_owner_count
```

全历史普通KV通常R(x)=x；压缩KV按真实产出规则，不能无条件假设floor/ceil。MLA不额外乘K/V的2；压缩记录覆盖原始token单位与kernel存储行数分别列出。泛化公式只能用于实现符合的分支。

分页用实际起止位置映射。共享边界页不能对prefix和chunk分别ceil再相加；若copy-on-write产生独立页，按真实副本计费。累计逻辑页差可用于归因，但不能替代有拷贝/alias时的物理页集合。

## 4. SWA至少分清三种集合

1. **命中边界所需尾窗**：按`find_longest_cache_hit`等实际规则查需要多少连续页；它不是整个prefix长度。
2. **prefix池全部保留窗口页**：从checkpoint边界、retention interval、分叉点、replay位置、实际产生/保存和淘汰情况得到唯一页集合S_prefix。多个边界尾窗可能重合。
3. **活跃执行页**：历史末尾保留窗口与未结算chunk的页集合S_active；已命中的尾窗可能同时在S_prefix中。

```text
S_swa_resident = S_prefix ∪ S_active
S_extra_prefix = S_prefix - S_active
```

在支持相应规则的版本中，活跃窗口admission上界为：

```text
n_active = ceil(min(max_model_len, W-1+extra+F)/block_tokens) + slack
```

该slack用于执行窗口跨页余量，不能作为命中checkpoint额外页直接套用。实际页集合未获得时可给上界，但不得声称所有层/组件上界同时实测达到。

若为归因把`n_active(F)-n_active(0)`记为chunk增量、`n_active(0)`记为历史基线，须明确是预算分摊，不是prefix cache确切页身份。不能据此声称整个prefix池只占这个基线。

读取运行版本的retention配置默认值及覆盖：它可能是仅语义checkpoint、周期checkpoint或密集保留，不能跨版本固定含义。即使仅语义checkpoint，也不能假设全池只有一个checkpoint。未知时给`M_active + cost(S_extra_prefix)`，不把未知项填0，也不把结果称为完整节点驻留总量。

## 5. state分别判定

- 固定ring/递推state：按实际tensor槽位，一请求一份；若不可prefix复用，prefix列标“不适用”，不在两批chunk中复制两份。
- 窗口式compressor state：按自己的逻辑block、窗口与F计算，不能套主KV压缩比，也不能沿用别的模型固定ring。
- prefix state checkpoint或恢复副本：仅在实现确实保存/恢复时计入，核查与执行state是否同一存储。
- FP32 state、BF16 scratch、Engram历史等与量化主KV精度分别确认；非KV持久buffer另列，说明是否已经从可用KV池预算扣除。

## 6. 异步默认与实际在途

从最终解析配置和源码分支确认async；未显式传参不表示关闭，兼容性检查也可能覆盖默认值。实际异步开启时使用异步作主结论。

```text
F_global_max = resolved_max_concurrent_batches * effective_batch_token_budget
sum(F_request) <= F_global_max
```

PP、runner和推测模式会改变批次数/位置数。对纯prefill的单请求，未结算输入不能超过该阶段尚未安全完成的输入位置数；P命中90%不代表能同时提供两个完整8K chunk。若有生成/推测位置，单独列入，不直接用剩余输入限制全部F。

D按各未结算step实际喂入token计F；无推测时可能每步1 token，有推测时不能照抄。不能把公共8K/16K预算作为每条decode请求的实际值。异步不把长期KV、prefix或固定ring整体乘并发批数；每个位置只计算一次。

## 7. PD释放：三种不同生命周期

| 机制 | 核查路径 | 计算规则 |
| --- | --- | --- |
| connector `delay_free_blocks` | `request_finished[_all_groups]`返回、scheduler合并、finished_sending/recving完成回报 | 完成但未释放的原页继续持有；通常是延长寿命，不是新副本 |
| 在途计算fence `defer_block_free` | consumer角色、并发批次数、last_sched/processed seq、deferred_frees | 即使connector返回False，也可能需等待旧写入安全结束 |
| prefix缓存驻留 | ref_cnt、hash、free queue与淘汰 | 请求引用释放后可能仍保留可淘汰内容，不等于不可用块 |

确认某layerwise push实现返回False时，其“完成后等待D拉取”持块项为0；不能把这一事实推广到所有layerwise connector、异常路径、MultiConnector的其他成员或计算fence。不能额外虚构一整请求的延迟KV副本。

layerwise传输与**物理buffer reuse**不是同一开关。若allocator把多层改为共享scratch，重新从最终布局计算slots、groups和pool成本；不可沿用全层独立缓存预算。传输缓冲/暂存副本按实际shape和生命周期另列。P/D双端只在时间重叠时相加，跨节点不能按逻辑相同prefix去重HBM副本。

## 8. 最终物理去重与结果名称

从组件记录账映射到最终group/tuple；同一full group里的主KV、索引和folded资源不能各乘一次完整pool单位成本。别名view只计一次，padding也只计一次。使用页身份应至少区分rank、pool与分配代次，不能跨节点或跨时刻仅按block编号去重。

已证明统一pool成本U时，可用以下口径；分桶成本不同则按真实分配分别加权：

```text
N_active：活跃请求页，跨请求共享页去重
N_delayed：完成后仍被connector/fence持有且尚未计入的页
N_extra_prefix：不在前两项中、仍有有效缓存内容的prefix页
M_held = U * |B_active ∪ B_delayed|
M_resident = U * |B_active ∪ B_delayed ∪ B_prefix|
```

N_extra_prefix可能可淘汰并可重新分配，不应全从admission可用量中扣除。M_resident为有效内容所占页的等价容量，不是启动预分配总量，也不是设备API的已分配内存；两者须另外说明。新请求边际分配按已有可复用页扣重，再考虑copy-on-write及恢复副本。

交付顺序：条件/版本 → Attention组成与owner → 逐组件prefix/chunk表 → 逐批F及checkpoint保留 → 物理页与padding → 释放生命周期 → 活跃持块、驻留量、边际量及日志对账。缺prefix保留页时，明确已给数字只涵盖哪一项。

## 有边界条件的检查算例（不是模型默认值）

某已知几何：W=128、block=128、40个独立SWA层、10个group、每记录544B、统一U=462848B，无推测额外尾部。若命中逻辑要求最近W−1个历史token，在对齐边界每组只需1块：

```text
单边界尾窗有效页 = 40*128*544 = 2785280 B
对应pool成本 = 10*1*462848 = 4628480 B
```

活跃admission在F=0时可能给每组2块，不能把该余量当成checkpoint精确成本。若多个checkpoint的尾窗页重合，不能直接乘checkpoint数；即使这个几何成立，也不能断言整个prefix池仅驻留上述一组尾窗。

另一个调度检查：输入131072、90%连续命中、按128对齐时P=117888，剩余13184=8192+4992。假设PP=1、实际两批在途、公共batch预算8192，单请求纯prefill的F包络为13184而非16384。其真实是否达到，仍需调度日志验证。上述数值只验证方法，不替代任意模型/版本的源码布局。
