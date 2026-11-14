---
name: activation-hbm-estimation
description: 根据模型配置、vLLM/vLLM-Ascend 执行源码和运行日志，估算推理中除权重和 KV cache 外的逐卡 HBM 峰值，覆盖 Attention/Indexer、Dense/MoE 激活、AllGather/AllToAllV、workspace、HCCL、图池与运行时开销。用于比较通信分支、定位阶段峰值和补齐非权重非 KV 显存预算。
---

# 推理激活与运行时 HBM 峰值估算

交付可复算的逐rank、逐阶段非权重非KV内存账，给最大单卡和峰值位置。用户所称“激活显存”通常指除权重和KV外的全部设备占用；先明确口径，再细分纯激活、workspace、通信、图池、持久非KV状态、运行时和allocator额外占用。训练保存激活/反向传播不套用本推理账。

## 边界与证据

- 排除所有权重及其转换/聚合副本、常驻KV池；计入临时K/V投影、KV gather/打包/反量化staging、非KV索引和共享缓冲。参数名含KV不足以判断是否排除，按用途、存储归属和生命周期分类。
- 用户只问纯张量峰值时可缩小范围；要求“除此以外所有显存”时覆盖通信、图池、非PyTorch、持久缓冲和allocator余量，不能只交付FFN公式。
- 从现有启动命令、环境变量、config、量化描述和日志取值；默认值沿实际版本覆盖链确认，未填写不等于关闭或为零。找不到的关键项保留符号和获取方法，不臆造实测值。
- 固定实际vLLM/vLLM-Ascend、TorchNPU/op-plugin、CANN、硬件和模型revision，记录相关未提交修改。部署文档对应旧镜像、分析使用新源码时，不能声称已证明旧镜像行为。
- 不重复索取已知参数。用户更正token、并行或dtype后立即更新有效输入和全部依赖结果；旧数字只作标明失效条件的对照。
- 用户要求自行选输入时，先找仓库内完整部署配置及实际模型元数据，说明为选取的参考场景。只读分析不授权改服务配置或执行昂贵profiling。

完整HBM估算由总入口传入共享默认负载：无明确实际值或用户要求时，序列参考长度128K=131072 tokens、连续prefix命中率90%。激活计算仍使用本轮实际chunk/query token或部署batch预算，不能把131072直接当每轮输入，也不能把全部激活乘10%；命中历史、窗口和workspace仍按实际路径保留。单独调用需要历史长度且无明确值时采用相同默认并标为估算假设。

## 固定负载和并行作用域

先填：模型/硬件/版本、P/D或混部角色、每DP调度token预算、实际本轮token、max_num_seqs、实际上下文组合、Attention TP/CP、有效EP、专家内部TP、PP、激活/通信dtype、图模式/档位、MTP/draft和重叠配置。

- 明确8K是8192还是8000；“单DP8192”不能当全实例额度。多请求共享本轮预算，不能给每请求各分配8192。
- Prefill使用本轮chunk token；decode按实际执行位置，投机验证/MTP按真实布局计入。max_model_len不是本轮query token，但会影响workspace和metadata。
- 从有效进程组和构造器核对EP；常见无额外CP的vLLM拓扑为`EP=DP×TP`，不能推广到PP/PCP/外部DP等所有拓扑。EP16不能唯一决定TP/DP。
- 只给EP而缺TP/DP时可列少量条件分支，不能默选并声称确定部署。名义TP与Attention头切分、MoE token分片、专家内部TP分别核对。
- 全实例token去除TP复制。每DP独立token数为T_dp时，`T_unique=T_dp×DP`；SP成立且均分时，本卡原始MoE token为`T_dp/TP`。
- 用户要求专家强制均衡时，`N_recv=T_unique×top_k/EP`，每专家`T_unique×top_k/E`，再按算子整除/对齐；发送量还取决于源rank token分布，接收均衡不自动保证发送均衡。
- `enable_balance_scheduling`不证明专家路由均衡；profile的随机矩阵+argsort不等于严格均衡，且自身产生临时内存。理想均衡与真实profile额外开销分开。

## 以物理存储和生命周期取峰值

沿“模型注册/patch → prepare → router/dispatch → 算子包装器分配 → kernel/workspace → combine/finalize → 释放”追踪。

每个storage记录：唯一身份、rank、实际分配shape、dtype/scale、物理padding/格式、owner/alias、创建/释放阶段、来源和确认状态。

```text
M_r(t) = sum(bytes(s) for s in 本rank时刻t存活的唯一且纳入范围的storage)
M_peak = max_r max_t M_r(t)
```

- 有效计算行数不等于分配行数。检查active_num、capacity、padding、图bucket和empty/apply_tensor形状；group_list限制计算不自动缩小输出。
- view/split/transpose按backing去重；切出2048行可能仍持有8192行storage。cat、clone、contiguous、量化和格式转换检查旧新共存。
- 追调用栈/dataclass引用，不以局部变量赋值认定旧张量释放。异步collective、多流和record_stream可能延长寿命，核对fence后才可复用。
- 普通串行推理不按层数乘所有中间激活；跨层持久缓存、图固定地址、PP在途微批和并发流水另核实。
- 同时存活才相加，顺序阶段取max。图池或通信池已有的同一storage不逐tensor再加一次；各阶段独立峰值不能机械相加。
- Attention若包含hidden/residual、公共buffer，MoE也按同一边界补齐后比较，不能以全账对比三份输出小计来宣布全局峰值。

## 按模块核算

按需读取[模块与源码追踪](references/module-accounting.md)。

1. Attention/Indexer：头分片或token分片、MLA吸收维度、Query/RoPE拼接、Indexer query/scale/top-k、KV staging、latent输出、V/O投影、调用方预分配输出、共享索引及workspace。融合Attention不默认物化完整Q×上下文score，也不忽略tile workspace。
2. Dense/MoE：gate/up、融合SwiGLU/quant、专家输出和workspace按实际dtype。W8A8不代表全部中间结果INT8；融合可能消除BF16 gate/up却保留较大INT32工作区。
3. AllGather：核对gather组/SP、先量化还是先gather、router logits、路由容量与本地有效专家数、GMM输出行数和全量形状combine。
4. AllToAllV：源端top-k展开、量化旧新共存、发送/接收、重排、专家计算、逆重排和反向通信。旧输出/发送/接收的共存需源码证据。
5. 共享专家、logits/sampler、MTP/draft、多模态实际分支另列。词表gather、FP32转换、只采样末位置或全token logits影响峰值；共享专家串行/多流重叠不能混用。

## 通信、图池与设备占用

读取[HCCL、图池与测量](references/runtime-pools.md)。

- HCCL按实际独立通信域、每域配置、收发规则累计；HCCL_BUFFSIZE不是整卡总额，也不是每peer各一份。考虑物理复用和惰性初始化，不按Python进程组对象数量相乘。
- TP/DP/EP、MC2、EPLB、world/内部world、P/D等仅计实际分配项；HCCL、MC2 window、symmetric/SHMEM池、普通张量buffer分别识别并去重。
- Prefill切到AllToAllV不自动移除decode/初始化保留的MC2资源。环境变量400不证明每个域都用400，每域options可以覆盖。
- 图池核对全部实际capture后的持久增量。FULL_DECODE_ONLY下prefill eager仍与decode图池共存；全局池不等于“每档一份”，也不保证所有图/层完全复用。
- 无日志/设备时可算已知图内buffer，并保留图metadata/池扩展未知项G；不得将G取0、固定若干GiB或按参数量/层数/prefill token线性猜值。
- 区分逻辑张量、allocated、reserved、设备free差分。reserved包含allocated，不能相加；non_torch已包含HCCL/ACL时不再加同一通信开销。安全余量与实测值分列。

## 异步跨批占用

确认async有效值、B个最大在途批次、每DP公共token预算和target/draft执行布局。逐storage记录当前batch及前序未完成batch保留的输入、输出、hidden/draft state、采样metadata、通信staging与图descriptor buffer；以真实跨批存活差额求异步非KV增量。两批在途不证明两次forward同时计算，不把全部激活、通信池或图池乘B。异步增量按A/G/C/P/R实际owner归类，只计一次。缺少B、shape或释放时点时，输出条件式并标“尚未完成定量核算”；局部buffer已算不代表完整异步峰值。

## 输出与复算

每个激活/workspace、图池、HCCL域、其他持久buffer、runtime和异步跨批模块均用计算表列shape/dtype、capacity、公式与数值代入、整数bytes、MiB/GiB、存活阶段/去重及状态。中间结果不能只写散文或代码块；未知storage仍列条件式和缺项。

单独调用时先给配置、模块计算及跨批存活分析，最后给分类汇总表，分别列激活/workspace A、图模式/图池 G、HCCL buffer C、其他持久缓冲/独立通信池 P、运行时与分配器额外量 R，以及同阶段小计/完整峰值或未知式。图模式需同时标明target/draft及占用证据。局部持久buffer不能填到A行并冒充完整激活；G/C未知必须保留在末尾汇总表。`non_torch`未能分解时注明覆盖类别，只计一次。

由 `model-hbm-budget` 编排时按 A/G/C/P/R 分别交接，不将五类折叠成一个“非KV”数；完整报告由总入口统一汇总，专项仅提供各模块计算、同存关系和证据。末尾汇总重复完整模块名称、配置/阶段、变量定义和异步核算状态，不使用上下文指代。计算过程展开：

1. 参数/来源表：用户值、实际默认、参考场景、显式假设和未知；明确rank/芯片/节点作用域。
2. 阶段表：`阶段 | 同时存活storage形状/dtype | 公式及数值代入 | bytes/MiB/GiB | 来源/未知项`。大项保留整数bytes，最后换算；GB=10^9、GiB=2^30。
3. 持久项：通信、图池、共享buffer、非PyTorch运行时、allocator额外占用分列，说明与阶段表去重关系。
4. 总峰值：全部必要项和阶段覆盖后才称完整估算；否则给“已核算阶段小计 + G/U”等条件式和未覆盖项。未知不是0，静态候选不是实测或保证上界。
5. 指出缺失值的具体日志/符号/测量方式；继续完成可独立做的静态核算，不因无法上板反复停在同一句免责声明。

已有storage清单可用零依赖脚本汇总：

```bash
python3 scripts/calculate_peak.py --input /path/to/storage-ledger.json
```

输入与验证边界见[storage账本](references/ledger-format.md)。脚本只按已提供存储取峰值，不自动解析模型、推断alias或发现漏项。

GLM-5.3、单DP8192、TP4/DP4/EP16的分支与Attention历史算例见[版本化算例](references/glm53-example.md)。仅在执行路径和分配规则相符时复用算术，不把其中0.56、3.72、1.66或0.83～0.90 GiB当通用默认值。
