# personal_agent

个人 Agent Skills。

## 默认推理后端背景

用户未另行指定、且实际部署证据未表明其他后端时，HBM、权重、KV Cache、激活及运行时占用分析默认采用 **vLLM + vLLM-Ascend（Ascend 执行路径）**，不再重复询问后端名称。用户显式指定的其他后端或已确认的实际部署优先；发生冲突时说明差异，不混用执行路径。

默认后端只确定分析路径，不确定硬件代际、vLLM/vLLM-Ascend 版本、KV dtype、并行布局、chunk、异步、MTP或缓存池配置；这些仍按实际代码、配置和日志取证，关键未知项保留条件并集中补问。优先追踪配套vLLM与vLLM-Ascend的注册、替换及最终生效路径；只找到上游vLLM实现不代表Ascend插件沿用同样布局。模型官方参考实现仅用于解释结构或明确标注的辅助对照，不能替代默认后端的物理容量账，也不能因缺参数而静默切换到参考实现或BF16布局。

## 完整 HBM 估算

当请求是“估算一个模型的 HBM/显存占用”时，使用 [model-hbm-budget](skills/model-hbm-budget/SKILL.md) 总入口，联动下面三个专项 skill。完整方法见 [HBM 占用指南](skills/model-hbm-budget/references/hbm-guide.md)，仓库内自动路由见 [AGENTS.md](AGENTS.md)。总入口只负责编排、统一配置和去重合账，三个专项继续维护各自算法。

| Skill | 用途 |
| --- | --- |
| [kv-cache-budget](skills/kv-cache-budget/SKILL.md) | 根据模型结构、vLLM/vllm-ascend 源码与运行日志，核算 KV Cache、滑窗、压缩器状态及 PD/异步在途容量预算。 |
| [weight-hbm-estimation](skills/weight-hbm-estimation/SKILL.md) | 根据模型 checkpoint 与 vLLM/vLLM-Ascend 实际代码，逐模块核算并行切分、共享专家 DP、Engram offload、量化转换及权重副本的 HBM 占用；关键未知配置先询问。 |
| [activation-hbm-estimation](skills/activation-hbm-estimation/SKILL.md) | 根据推理执行源码与运行日志，核算除权重和 KV 外的激活、Attention/Indexer、MoE AllGather/AllToAllV、workspace、HCCL、图池及运行时逐卡峰值；按存储生命周期去重并保留未知项。 |

用户未指定且没有实际负载取值时，默认按序列参考长度 **128K=131072 tokens**、连续前缀命中率 **90%** 估算，明确标注默认假设；实际取值或用户要求优先。命中前缀按后端粒度对齐，命中率不直接乘减整池或整模型显存。

每个模块的中间计算结果均用表格输出，包含参数、公式与数值代入、bytes/GiB、作用域及状态。

默认输出顺序为：**参数与证据 → 各模块计算过程 → 异步调度增量 → 逐阶段合账与容量检查 → 末尾总体汇总表**。末尾汇总表分别列出 KVCache、权重、激活/workspace、图模式/图池、HCCL buffer、其他持久缓冲/通信池、运行时与分配器额外量，以及总计和条件余量。未知项保留符号，不填0；完整格式见[报告模板](skills/model-hbm-budget/references/report-template.md)。

调用流程为：一次收集共享配置 → 三个专项形成分账 → 根据非KV预算求实际KV池 → 计算异步增量 → 同rank同阶段去重合账 → 末尾汇总。读取skill不等于已经完成相应模块的计算。

例如：“使用 $model-hbm-budget 估算模型 X 在这份部署命令下的 HBM，先展开公式、数值代入、异步在途和跨批占用及加载/运行峰值，最后按 KVCache、权重、激活、图模式、HCCL buffer 等给单卡总体汇总表，写明模块名称和配置，不使用上下文指代。”

在其他工作区使用时，将 `skills/` 下四个目录（总入口与三个专项，含 references/scripts）一起安装到 Agent 的 skills 目录；Codex 通常为 `~/.codex/skills/`。安装后可显式调用 `$model-hbm-budget`，也允许按描述自动选择；自动发现取决于宿主的 skill 加载机制。仅更新 GitHub 不会自动安装到本机或其他会话。只需单项时直接调用对应专项。
