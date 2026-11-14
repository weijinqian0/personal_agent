---
name: weight-hbm-estimation
description: "根据模型配置、checkpoint 元数据及 vLLM/vLLM-Ascend 的实际加载代码，估算各模块、各 rank 的权重 HBM 占用。用于权重显存、TP/EP/PP/DP 切分、共享专家 DP、Engram offload、量化格式转换和权重副本核算；影响结果且无法确认的配置必须询问。"
---

# 权重 HBM 占用估算

目标：给出指定部署路径下的逐模块、逐卡常驻权重及量化元数据容量，并解释影响结果的并行切分、offload 和运行时格式变化。适用于推理；训练梯度、优化器状态，以及 KV/激活总容量不属于本 skill 的权重总数。

## 不确定项必须先询问

这是用户明确要求：**涉及估算结果的并行配置或实现行为，若在现有对话、实际部署配置和 vLLM-Ascend 的有效代码路径中无法确认，必须询问用户后再给出依赖它的最终总数。**

- 先检索本地部署命令、配置、环境变量、HF config 和代码，不重复询问已明确的信息。找到默认值后，检查实际配置是否覆盖；可以确认的有效默认值需在结果中标明。
- 同时存在多条实现分支、只找到上游 vLLM/官方示例而没有 Ascend 生效证据、代码与 checkpoint 不匹配，或底层算子物理布局无法推断时，标为“待确认”。不能把“没搜到”当作未启用、零字节或与参考实现相同。
- 每次聚合询问决定结果的少量问题，说明各选项影响哪个模块；例如“EP32 是有效 EP 组大小，还是 DP32×TP2？共享专家 DP 开关是否开启？”
- 等待回答期间继续完成无依赖的代码核对和已知分项。可以给出明确标注的条件分支或已确认小计，不能自行选一个分支、声称完整估算，或把等待超时当作确认。
- 用户也无法提供时，交付已确认小计、符号化未知项及获取方法；不要声称得到精确 HBM 或实测结果。

## 输入与证据

1. 定位用户指定或工作区附近的 vLLM、vLLM-Ascend 仓库，遵守各自 AGENTS.md。记录 HEAD、相关文件修改和内容哈希；只读核对，不改用户源码。已提交 HEAD 与未提交工作区若影响容量，分别报告。
2. 定位模型 revision、HF/config、safetensors index 和全部 shard header；优先使用 header 的 dtype、shape、data_offsets 和 scale，而非模型名的 B/T 标签。没有必要时不下载完整权重、不执行远程模型代码。
3. 确认目标硬件、model dtype、原生/ModelSlim 等量化路径、模型注册及 plugin 替换、模型加载器和运行阶段。不能把 CUDA 路径、Ascend 路径与官方简化参考实现拼成一个方案。
4. 确认 TP/DP/PP/PCP/DCP、有效 EP、专家内部 TP、共享专家 DP、独立 MLP/O-proj/词表并行组、DSA-CP、mix_placement、EPLB 冗余、Engram offload，以及视觉/MTP/DSpark 是否实际加载。仅核对与目标模型有关的配置。

## 逐模块核算

需要模块清单及源码定位方法时，读 [模块与代码核对指南](references/module-accounting.md)。DeepSeek V4.1 的历史校验样例见 [版本化算例](references/deepseek-v41-example.md)；样例数字不可直接当作当前部署结论。

- 先建立不重叠的权重分类：embedding/head、注意力与公共低秩投影、Indexer、compressor、Dense FFN、router、路由专家、共享专家、Engram 大表与附属投影、norm/连接参数、实际加载的附加模块。
- 跟踪“模型注册 → 构造器 → Linear/FusedMoE 的底层并行组选择 → quant method → create_weights → weight loader → process_weights_after_loading → 执行路径”。高层 ColumnParallelLinear 或 disable_tp 的名字不足以证明最终切分。
- 为每个张量记录：全局形状/格式、所在 PP stage、有效切分组与轴、复制范围、本卡 dtype/shape、scale/zero-point、padding、offload 目标、额外副本、代码位置和确认状态。
- 对每个 rank 求和，报告最大单卡和不均匀分布；不要只给平均值。MoE 驻留按本卡实际专家集合计算，不能按 top-k 激活比例缩小；单步读取量也不能冒充驻留量。
- 根据实际 storage/别名去重：view/transpose/共享参数不自动增加字节；clone/contiguous/format_cast 需追踪旧存储是否保留。共享池按唯一 key 与生命周期累计，不能机械乘层数。
- 将 checkpoint 字节转换为加载后字节：量化 scale 展开、实际 dtype 升级、预转换副本、执行打包、冗余专家和共享缓冲分别加减。转换中旧/新共存的峰值另列。
- Offload 只扣除真正卸载的张量及元数据。投影/gate、设备缓存、地址表与 lookup/staging buffer 分别核对；UVA 设备地址不意味着整表占 HBM。未发现缓存实现不能臆造固定百分比缓存。

## 输出与验证

单独调用时先给权重汇总表，列常驻权重、量化元数据、额外权重副本、已知小计/未知补充项及加载转换峰值的状态；表后再给配置、逐模块公式、数值代入和来源。关键未知未确认时首表给条件式/已知小计，再集中提问，不给依赖它的完整总数。

由 `model-hbm-budget` 编排时，只交接 W、转换副本、阶段/rank与证据状态，非权重buffer交接给激活专项；完整报告采用总入口首表与详算顺序，不重复输出独立报告。正常交付包括：

1. 配置摘要：版本、工作区差异、有效并行组、量化路径、offload、实际加载模块及明确排除范围。
2. 分项表：`模块 | 实际放置/切分 | 单卡字节/GB/GiB | 格式或副本变化 | 源码证据/待确认项`。
3. 汇总：加载后常驻权重、额外权重池、非权重持久缓冲、加载/转换峰值分别列；注明 GB=10^9 bytes、GiB=2^30 bytes。给配置变化的增减值时，固定其他条件。
4. 代码和算术核对证据、未覆盖项及精度边界。静态张量字节不是 allocator reserved；未知的 NZ storage padding 不能假定为零并声称精确 HBM。

复杂重复计算可保存参数化脚本与机器可读结果，使用当前 config/header 驱动并检查分类去重、分片完整性和最大 rank；避免复制只适用于某个快照的常数。若用户已授权且具备设备条件，可按相同配置测量加载前后 allocated/reserved；区分权重差分、加载峰值、warmup/图捕获和 KV 分配。没有实测条件时明确说明，不因无法上板而伪造验证。
