# 单请求 KV Cache 固定核算流程

适用于要求具体KV占用或物理池需求的请求。按八步推进；遇到未知时继续完成无依赖项，保留公式与缺口，不能通过静默改场景得到唯一总数。模型专有布局数值来自当前证据，不作为此流程的默认值。

## 1. 锁定身份、口径与输入来源

先建立参数表：值、confirmed/assumed/unknown、证据、影响。输入K/k显式声明1000或1024 tokens，保留整数bytes并分别标MiB/GiB与十进制GB。至少覆盖模型revision，后端commit与相关dirty diff，硬件/设备能力，生效patch，KV/索引/state各自dtype，实际block，prefix开关和命中粒度，batch上限，异步批次数，MTP/DSpark方法/owner/K和extra retention。

默认后端vLLM+vLLM-Ascend只是追踪入口，不确定上述参数。使用本地config、解析后启动配置和日志，再沿注册和替换确认实际路径；官方参考实现不替代Ascend物理账。只对缺失L/h使用SKILL.md约定，不补造block、C8或DSpark配置。

产物：参数/来源表、计算范围与结果口径。旧报告仅在模型、代码、硬件、dtype、block、hybrid、draft相关身份匹配后复用；路径相同不证明版本相同。

## 2. 定义唯一执行快照

```text
P_target = floor(L*h)
P = g*floor(P_target/g)
S = P+Q
F = sum(C_i)
H = P+Q+F
```

g取实际协调器与各group约束，不等同base block。展示目标命中率和P/L。命中历史仍驻留，不能把总KV乘(1-h)。若h是服务平均值，不能自动当作每请求连续前缀。

纯prefill必须满足Q+F<=L-P。L-P是全部剩余输入；当前C_i按实际调度取值。max_num_batched_tokens只是该batch公共上限，单请求占满额度需标条件。逐批标识batch_id，不默认两批，不给剩余输入不足的请求添加两批完整chunk。有生成或lookahead位置时单列来源。

未知chunk仍写C_i公式，不退回F=0。辅助展示结算后静态量时明确其用途，不取代执行预算。跨快照比较先统一P/Q/C_i；不能以F=0对比F>0的差额直接声称全部是漏算。

产物：L/P/Q/C_i/F/H快照表，未处理位置数和条件场景边界。

## 3. 从实际构造遍历每个缓存owner

逐种Attention遍历，不因名称为MLA就忽略SWA/Indexer。清单包括：主K/V或共享latent、压缩记录、Index K/scale、真正独立folded副本、SWA、主compressor、index compressor、递推/卷积状态、实际target/draft/MTP/DSpark owner。不存在的组件按构造证据标不适用。

对每个owner记录数量、target/draft角色、shape、dtype、scale/padding、压缩率、窗口、logical/storage block、backing/alias。K/V共用latent不乘2；主压缩器输出alias不另算长期副本；Query头数不自动成为KV存储头数。state逐项取dtype，不能随KV量化统一缩小。

投机方法未知时列未知owner/额外项，不把关闭当默认，也不凭config存在附加层就声称实际启用。附加层数、推测token数和state版本数分别取证。

产物：owner inventory，后续每张组件账均须覆盖inventory，不缺行。

## 4. 确认写入、恢复、保留、释放、去重

逐owner填写五项生命周期证据。沿kernel输入/slot_mapping、cache manager、checkpoint匹配与释放调用判断，不能只读tensor shape。

- 长期记录：确认实际R(x)、压缩边界及是否保留全历史。
- 滑窗分页：分别确认命中边界尾窗、全部保留checkpoint、活跃尾窗、未结算位置与延迟页；每种state用自己的W。
- 固定ring/递推槽：按执行槽/版本数计费，不随C_i盲乘；新增chunk可能不新增槽，但既有槽容量不能写0。
- 恢复、活跃、额外保留集合按物理页身份取并集；复制/CoW和回滚另计。admission余量不是精确checkpoint集合。

产物：生命周期表。缺证据项保留unknown，不将额外prefix/投机保留填0。

## 5. 每个组件做prefix/Q/逐chunk记录分账

令T_i=S+sum(C_1..C_i)，T_0=S：

```text
prefix_records = R(P)
settled_records = R(P+Q)-R(P)
chunk_i_records = R(T_i)-R(T_(i-1))
```

长历史各列之和必须等于R(H)。压缩组件不得分别floor(C_i/r)。scale、RoPE、Index等按真实记录布局计字节；窗口state不套长期R(H)守恒。

正文使用此表；同结构可合并owner数，所有适用列保留：

| 组件/owner数/dtype | Prefix P | 已结算Q | C1 | C2… | 活跃窗口/固定槽 | 额外checkpoint/投机保留 | 来源/状态 |
|---|---|---|---|---|---|---|---|
| 每个inventory owner | 记录、公式、bytes或未知 | 累计差分 | 位置数、累计差分、bytes | 逐批独列 | 自身生命周期 | 已知或未知，不省略 | confirmed/conditional/unknown |

## 6. 转换组件唯一页，列页内padding

展示每记录数据/scale/填充字节、页行数、覆盖原始位置数、实际页字节。有效记录和物理页分别核算；不要先把各chunk独立ceil再相加。

长期页可按累计位置的M(x)差分；更一般按物理页集合新增差集归因。跨列同页只计一次：新增有效记录不一定新增物理页。窗口计算还要纳入回收、checkpoint与延迟释放。

`ceil((W-1+extra+F)/logical_block)+slack`只用于代码确认采用此规则的窗口管理器，不能当普适公式。标明admission上界或实际持块，不把各组件不能同时达到的上界称为运行峰值。

产物：逐组件分页账，`component_page_bytes`。这一步完成不代表物理池完成。

## 7. 从group遍历到唯一物理backing

读取[物理布局核对](physical-layout-validation.md)，从当前最终spec清单追到生效分配器，而非仅引用Spec。

1. 按真实分组键和拆组算法形成group；记录approximate GCD/层数拆分结果。
2. 逐group列page-size buckets、owner数和原始/填充页大小。
3. 确定normal tuple slots与draft/MTP尾槽；明确空层槽、缺失bucket仍预留的空间。MTP改变布局时重新分组，不能沿用无draft槽数。
4. 沿offset、layer_stride、block_stride追到唯一backing分配ID，判别跨group alias与独立block ID。多个descriptor的size可能指向同一backing，不能直接相加。
5. 依实际分配器计费：独立页、多池、固定ring分别核算；只有确认shared-tuple且无额外alignment时才用U=slots*sum(bucket_bytes)。同组main/index共享一次完整U，不各乘一次。
6. 把P、Q、各C_i、窗口和额外保留的唯一group块映射到成本，保留组件页到物理布局的桥接差额。

正文补两张表：

| group/成员 | logical block | bucket B/owner数 | normal/尾部slots | 唯一块归因P/Q/C_i/窗口/extra | 每分配单位B | 条件/来源 |
|---|---|---|---|---|---|---|

| 同一身份与快照 | 组件页bytes | group/tuple额外bytes | 已知页映射bytes | 未闭合保留项 |
|---|---|---|---|---|

桥接只比较同dtype/block/draft和同快照；不能把配置变化计为padding。未知保留项存在时已知映射仍是小计，request_pool_need_bytes=null；不能用“条件”标签或附件绕过完整账。

## 8. 对齐三条成本路径再汇总

分别追踪allocator的backing成本、admission的请求计费、capacity的预算分母。核对相同owner/组数/slots，特别检查MTP尾槽和边界余量。冲突显式列期望/实际、代码分支及差额；不能只选择较大数字就称验证完成。

有日志时，仅在同池、同分配范围核对唯一registered bytes/num_blocks；排除重复alias描述符和独立通信buffer。日志比值吻合不证明版本一致或request生命周期完整。

结果按“prefix→Q→窗口/固定态→逐批新增→额外保留”展示，再列五种口径。请求块是在池中消耗空间，不加到预分配池之上。静态数值不填observed_peak_bytes。

具体物理池需求通过[结构化交付门禁](release-gate.md)后由入口发布；成本冲突/未知生命周期时保留未完成和已知小计。符号公式不等于异步已完成定量核算。

## 发出答复前的检查

- 主文是否有所有owner的P/Q/各C_i分账，而非只附文件或总数？
- 是否误把L-P当单chunk，把batch上限当实际调度，或用F=0替代执行阶段？
- 是否覆盖所有会写入的SWA/state，区分固定槽和分页state及各自dtype？
- 是否有prefix恢复/保留/活跃页的并集证据，未知没有当0？
- 是否从组件页继续算group、bucket、空槽与MTP尾槽，并对alias去重？
- 是否统一前后数值身份与快照，并完成allocator/admission/capacity核对？
- 未完成时是否明确request_pool_need_bytes=null，且没有把小计写成最终HBM？
