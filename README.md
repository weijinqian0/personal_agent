# personal_agent

个人 Agent Skills。

| Skill | 用途 |
| --- | --- |
| [kv-cache-budget](skills/kv-cache-budget/SKILL.md) | 根据模型结构、vLLM/vllm-ascend 源码与运行日志，核算 KV Cache、滑窗、压缩器状态及 PD/异步在途容量预算。 |
| [weight-hbm-estimation](skills/weight-hbm-estimation/SKILL.md) | 根据模型 checkpoint 与 vLLM/vLLM-Ascend 实际代码，逐模块核算并行切分、共享专家 DP、Engram offload、量化转换及权重副本的 HBM 占用；关键未知配置先询问。 |
| [activation-hbm-estimation](skills/activation-hbm-estimation/SKILL.md) | 根据推理执行源码与运行日志，核算除权重和 KV 外的激活、Attention/Indexer、MoE AllGather/AllToAllV、workspace、HCCL、图池及运行时逐卡峰值；按存储生命周期去重并保留未知项。 |
