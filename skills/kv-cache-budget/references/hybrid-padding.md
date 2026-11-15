# Hybrid 与 padding：从组件账到物理池

在同一请求包含全注意力与 GDN/Mamba、SWA、不同压缩率、独立 Indexer 或 MTP/draft 时读取。本节也适用于单一 Attention 使用异构页或共享 tuple 的后端。模型名称包含“hybrid”不证明分配器使用统一页；最终布局以生效源码为准。

## 先确定交付口径

用户只问“KV 占用/容量/预算”，没有明确限定有效载荷时，继续完成物理池映射；不要自行把任务缩成“有效槽位预算”，再把 hybrid/padding 留成最后一句补充。可先交付已知组件账，但结论应标“物理池需求尚未算完”，`request_pool_need_bytes=null`，并列尚缺的布局证据。

仅当用户明确只求有效量时，可以停止于该范围；仍需说明它不含哪些物理填充。不使用固定百分比、“少量 padding”或经验余量补成总数。缺证据不代表 padding 为零。

## 必须分别核算的三层差额

| 层次 | 要取证的量 | 容易漏算或重复的项 |
|---|---|---|
| 记录到原生页 | 记录字节（含量化 scale/元数据）、storage rows、实际页行数、部分尾页 | 量化 scale 是记录成本；不足一页的空行是分页填充；各 chunk 用累计页差分，不能分别 ceil 后相加 |
| 原生页到 allocator 页 | 原生页 bytes、最终 block 覆盖范围、padded page bytes、alignment、block/layer stride | GDN 大状态可能使 GQA 页扩大、逻辑 block 重设或 stride 留空；确认是哪条分支，不能直接取 max 后自称完成 |
| allocator 页到共享池 | 最终 group 拆分、page-size bucket、normal/尾部 slots、缺失 bucket、独立 block ID、唯一 backing | 为均衡组宽预留的空层槽、整块缺失的 bucket、MTP 尾槽，以及不同组用同一池而产生的稀疏利用率 |

必须追到最终覆盖链：CLI 初值 → 模型/硬件校验 → cache spec → group planner → allocator → reshape/stride → kernel 读写。不能只用 CLI block 或 Spec 的原始页大小。若页大小/逻辑 block 被覆盖，重新计算 prefix 对齐、P/Q/C_i 页数与状态持块，不能只给旧结果补一项 padding。

`component_page_bytes` 应明确包含到哪一级页内填充；共享池空槽另归因。一个 padding 已进入 page bytes 或 pool unit 后不能再追加同一份成本。

## GDN/Mamba + GQA：状态份数不是物理池成本

分别保留 GQA 历史记录和 GDN 的 recurrent/conv state owner。FP8 KV 不表示 FP32 SSM、BF16 conv 或状态 checkpoint 被同步量化；MTP draft 还可能使用不同 cache dtype。

状态份数先从实际 `none/align/all` 分支与生命周期推导。若某版本 align 的规则是 `2 + speculative_blocks + prefill_checkpoint_blocks`，只能把它标为该版本的请求保留/admission 规则；不能泛化为所有模型的固定“五份”，也不能据此断言全部 prefix checkpoint 只有这几份。

对 prefix 恢复、活跃执行、MTP 验证/回滚、内部 checkpoint 和额外在途按唯一状态页取并集。单份状态 bytes × 预留份数得到原始状态组件预算；下一步还必须用 padded page、group block 消耗与 backing 映射定价。

尤其禁止将“完整 GQA KV + 若干份原始 GDN state”作为 hybrid 的物理总量。即使状态不随历史 token 数线性增长，共享池仍可能为 GQA 的每个独立 block 预留状态槽或空位；这种放大取决于 allocator，不能仅由状态 tensor shape 判断。

## 两类布局的不同计费

**独立页/独立池**：按已去重的物理页数 × 该池实际页字节求和。各池可以有不同 block 数、alignment 与 admission 规则，不能套共享 tuple 的槽数公式。

**已确认的统一页共享池**：若所有 slot 的物理页都是 D bytes，最终 pool 宽度为 S，且不同 group 消耗独立 pool block ID，则一个 pool block 成本 `U=S*D`，请求成本 `sum_g(unique_blocks_g)*U`。这里的 S 必须来自最终分组，而非原始模型层数；不同组的视图可复用相同槽位，但它们分配的 block ID 未必相同。

**已确认的多 bucket 共享 tuple**：只有源码确实完整预留每个 bucket 且无额外 alignment 时，才使用 `U=S*sum_b(D_b)`。组内实际 owner 只填部分 bucket，不使 U 自动变小；MTP 尾槽缺失的 bucket 也可能仍付费。main/index 同组只乘一次 U。其他布局按真实分配映射，不能强套此式。

## 可复算的合成例子：两种 padding 都不可省略

以下是已给定最终几何的教学例子，不是 Qwen 或任意后端的默认值。假定两个组消耗互不重合的共享池 block ID：全注意力组2个 owner、持100块；状态组3个 owner、持2块。全注意力原生页4096B，状态原生页8192B；allocator 统一页8192B、宽度3槽，无其他 alignment。

| 同一快照成本 | 公式 | bytes |
|---|---|---:|
| 原生组件页 | `2*100*4096 + 3*2*8192` | 868352 |
| allocator 页对齐增加 | `2*100*(8192-4096)` | 819200 |
| 对齐后组件页 | `2*100*8192 + 3*2*8192` | 1687552 |
| 共享池空槽增加 | `(3-2)*100*8192 + (3-3)*2*8192` | 819200 |
| 唯一物理池映射 | `(100+2)*3*8192` | 2506752 |

直接把组件相加会给868352B，而已给定布局需要2506752B；只补页对齐仍会漏掉819200B空槽。该例放大系数不用于其他模型。若改用独立池、改变分组、引入MTP或量化后导致原生页变化，必须重放布局而非沿用本例比例。

## 正文应展示的桥接

| group/bucket | owner数与实际存储 | 逻辑/存储block | 原生页 → padded页 | normal/尾槽/空槽 | 独立持块及P/Q/C_i归因 | allocator单位bytes与backing |
|---|---|---|---|---|---|---|
| 每个最终组 | 包括state/draft与空bucket | 填生效值 | 列差额和源码 | 从分组输出复算 | 边界页只计一次 | 共享view去重 |

另列“有效记录 → 原生组件页 → 页对齐增加 → group/bucket/尾槽空位增加 → 唯一物理映射”成本桥接。将 checkpoint 额外保留、lookahead 和异步持块变化另行归因；它们是生命周期/持块差异，不笼统改称 padding。字段可在报告中展开，不必改变现有脚本 schema。

allocator、admission 和 capacity 必须使用同一布局身份和范围核对。若实际布局没有某类填充，应给出该项为零的证据；如果未知则保留未知，不能在结论写“总计X GiB，另加padding”。已知组件数字不得使用“实际占用/部署总量/完整峰值”的名称。现有 `audit_layout.py` 只验证调用者声明的受支持布局算术，不能代替源码重放或自动证明 checkpoint 完整。
