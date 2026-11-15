# HBM 估算请求的路由

## KV 部署报告的可执行门禁

具体KV部署需求报告使用[交付门禁](skills/kv-cache-budget/references/release-gate.md)及 `skills/kv-cache-budget/scripts/release_budget.py`。先验证逐组件状态生命周期和P/Q/各chunk证据账，再发布由入口生成的结果；没有当前账本绑定的复核、退出非零或result.status不是released时，只报告缺口与已知小计，不手写部署总数绕过。源码理解和review身份仍需真实独立复核；脚本和CI不保证任意自由聊天内容。概念或明确限定静态payload的请求不扩大范围。

## 默认推理后端背景

用户未另行指定、且实际部署证据未表明其他后端时，HBM、权重、KV Cache、激活及运行时占用分析默认采用 **vLLM + vLLM-Ascend（Ascend 执行路径）**，不再重复询问后端名称。用户显式指定的其他后端或已确认的实际部署优先；发生冲突时说明差异，不混用执行路径。

默认后端只确定分析路径，不确定硬件代际、vLLM/vLLM-Ascend 版本、KV dtype、并行布局、chunk、异步、MTP或缓存池配置；这些仍按实际代码、配置和日志取证，关键未知项保留条件并集中补问。优先追踪配套vLLM与vLLM-Ascend的注册、替换及最终生效路径；只找到上游vLLM实现不代表Ascend插件沿用同样布局。模型官方参考实现仅用于解释结构或明确标注的辅助对照，不能替代默认后端的物理容量账，也不能因缺参数而静默切换到参考实现或BF16布局。

背景要求：生成总体模型的 HBM 空间占用时，必须先分别计算各模块的 HBM 占用，建立可复算的分项账，再按同一配置、rank 和阶段去重汇总为模型总账。计算顺序是“模块拆解 → 各模块核算 → 去重合账”，展示顺序是“配置与证据 → 各模块中间计算过程 → 异步增量与逐阶段合账 → 末尾总体汇总表”；两者不能混淆。

用户要求估算模型完整 HBM/显存占用、判断能否装下、计算剩余 KV 空间或并发容量时，使用 [model-hbm-budget](skills/model-hbm-budget/SKILL.md) 作为总入口，读取并应用以下三个 skill：

1. [weight-hbm-estimation](skills/weight-hbm-estimation/SKILL.md)：加载后的权重与权重副本。
2. [kv-cache-budget](skills/kv-cache-budget/SKILL.md)：缓存组件、请求持块与预分配池。
3. [activation-hbm-estimation](skills/activation-hbm-estimation/SKILL.md)：激活、workspace、通信、图池和运行时。

最终答复先给配置、模块计算与异步调度分账，最后输出总体汇总表，按 KVCache、权重、激活/workspace、图模式/图池、HCCL buffer、其他持久缓冲/通信池、运行时与分配器额外量分别列行，汇总表中的数值由模块计算与逐阶段合账导出。采用[报告模板](skills/model-hbm-budget/references/report-template.md)，表中明确已知小计、条件总数或完整峰值；未知项不省略。不要把三个专项回答直接拼接，或将不同阶段/不同 rank 的分项最大值直接相加。

用户没有特别指定、且实际部署/负载证据没有确定取值时，序列参考长度默认128K=131072 tokens、连续前缀命中率默认90%，分别补齐缺失字段并标“用户约定的估算假设”。明确取值优先，不修改实际max_model_len或prefix配置；90%命中不代表KV/HBM减少90%，也不代表多请求共享同一物理prefix。

每个模块的中间计算结果必须用表格输出，表格包含参数/来源、shape/dtype或记录/页/块几何、公式及数值代入、整数bytes、MiB/GiB、rank/阶段和状态；KVCache、权重、激活/workspace、图池、HCCL、其他持久量、runtime和异步增量均不能只给文字或公式。总体汇总表仍放在末尾。

异步调度必须确认有效开关、最大在途批次、公共token额度、KV新增/延迟释放块及非KV跨批buffer。只有给出数值增量或有证据的零增量才能声明“已计入异步”；仅保留O/U一类未知式必须标为“异步占用尚未完成定量核算”。KV池内异步持块不重复加到整池HBM。

输出使用完整模块名、模型配置、角色/rank和阶段，避免“上述、同上、该部分、其余项、见前文”等上下文指代。公式符号需定义，最终汇总表重复必要定义与条件，使总体结果能够独立阅读。

统一配置、物理存储归属和阶段生命周期后合账；指南见 [完整 HBM 占用指南](skills/model-hbm-budget/references/hbm-guide.md)。缺少决定结果的配置时，完成无依赖分项并集中补问；未知不能记零。仅问权重、KV 或激活时使用对应专项 skill，不强制扩大范围。训练显存不套用这套推理账。

本仓库的路由在工作区范围内生效。要在其他工作区自动发现总入口，应将总入口和三个专项 skill 一起安装到对应 Agent 的 skills 目录；不能假定仅推送 GitHub 就会自动安装。
