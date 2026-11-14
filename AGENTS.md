# HBM 估算请求的路由

背景要求：生成总体模型的 HBM 空间占用时，必须先分别计算各模块的 HBM 占用，建立可复算的分项账，再按同一配置、rank 和阶段去重汇总为模型总账。计算顺序是“模块拆解 → 各模块核算 → 去重合账”，展示顺序仍是“总体汇总表 → 各模块中间计算过程”；两者不能混淆。

用户要求估算模型完整 HBM/显存占用、判断能否装下、计算剩余 KV 空间或并发容量时，使用 [model-hbm-budget](skills/model-hbm-budget/SKILL.md) 作为总入口，读取并应用以下三个 skill：

1. [weight-hbm-estimation](skills/weight-hbm-estimation/SKILL.md)：加载后的权重与权重副本。
2. [kv-cache-budget](skills/kv-cache-budget/SKILL.md)：缓存组件、请求持块与预分配池。
3. [activation-hbm-estimation](skills/activation-hbm-estimation/SKILL.md)：激活、workspace、通信、图池和运行时。

最终答复先给汇总表，按 KVCache、权重、激活/workspace、图模式/图池、HCCL buffer、其他持久缓冲/通信池、运行时与分配器额外量分别列行，再给各模块计算过程。采用[报告模板](skills/model-hbm-budget/references/report-template.md)，表中明确已知小计、条件总数或完整峰值；未知项不省略。不要把三个专项回答直接拼接，或将不同阶段/不同 rank 的分项最大值直接相加。

统一配置、物理存储归属和阶段生命周期后合账；指南见 [完整 HBM 占用指南](skills/model-hbm-budget/references/hbm-guide.md)。缺少决定结果的配置时，完成无依赖分项并集中补问；未知不能记零。仅问权重、KV 或激活时使用对应专项 skill，不强制扩大范围。训练显存不套用这套推理账。

本仓库的路由在工作区范围内生效。要在其他工作区自动发现总入口，应将总入口和三个专项 skill 一起安装到对应 Agent 的 skills 目录；不能假定仅推送 GitHub 就会自动安装。
