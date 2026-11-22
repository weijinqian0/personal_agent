# personal_agent

个人 Agent Skills。

## 模型 HBM 报告

[打开在线报告首页](https://weijinqian0.github.io/personal_agent/)

| 模型 | 场景与范围 | 在线报告 | HTML 源文件 |
| --- | --- | --- | --- |
| DeepSeek V4 Flash | A5、TP4×DP4（条件 EP16）；异步 KV ×2 预算，128K / 512K / 1M / 2M 长度对照。 | [在线阅读](https://weijinqian0.github.io/personal_agent/models/deepseek-v4-hbm-report.html) | [HTML 文件](models/deepseek-v4-hbm-report.html) |
| DeepSeek V4.1 Flash | 128K / 256K / 512K / 1M 显存分布，2M 容量外推；90% 目标连续 prefix 命中、8192-token chunk、Engram 主机内存 offload。 | [在线阅读](https://weijinqian0.github.io/personal_agent/models/deepseek-v4.1-hbm-report.html) · [长度对照](https://weijinqian0.github.io/personal_agent/models/deepseek-v4.1-hbm-report.html#length-memory) | [HTML 文件](models/deepseek-v4.1-hbm-report.html) |
| GLM-5.3 | 沿用 DeepSeek V4 的 A5、TP4×DP4（条件 EP16）、8192-token chunk 与 KV ×2 预算；78 层 MLA/DSA、1 层 MTP，A5 量化权重采用条件分账。 | [在线阅读](https://weijinqian0.github.io/personal_agent/models/glm-5.3-hbm-report.html) | [HTML 文件](models/glm-5.3-hbm-report.html) |

在线报告通过 GitHub Pages 发布，浏览器可直接渲染。HTML 文件包含内置样式与图表，也可下载后离线打开。报告属于静态条件分析，完整物理需求与设备峰值的缺项保留在各报告中；2M 对照仅为字节几何外推，不代表模型已支持 2M 上下文。

## KV Cache 报告门禁

具体KV部署报告增加[证据账与受控交付入口](skills/kv-cache-budget/references/release-gate.md)：校验源码文件指纹与版本、全部Attention组件的状态生命周期和prefix/Q/各chunk分账、物理布局与allocator/admission/capacity成本，并绑定独立复核记录。未通过时退出非零，报告部署需求为null；失败重跑覆盖旧成功报告。原有算术脚本继续保留，不再被视作完成证据核查。

```bash
python3 skills/kv-cache-budget/scripts/release_budget.py \
  --input budget-reports/<case>/ledger.json \
  --review budget-reports/<case>/review.json \
  --output-dir .budget-gate-output/<case>
```

使用[账本模板](skills/kv-cache-budget/references/ledger-template.json)开始。没有review时先省略`--review`，生成缺项报告及待复核请求；不能自动批准。仓库工作流 `KV budget gate / validate` 运行回归与已提交账本的重新校验，只在成功后上传生成产物。将其设为required check及配置独立PR复核，需要管理员另外启用仓库规则；新增workflow本身不会开启分支保护。

边界：门禁强制约束通过入口生成的报告，不能拦截手写聊天，也不能自动证明源码解释或复核者身份真实。现有模板与synthetic测试夹具不是实际部署证据。GitHub更新不会自动安装到本机Agent skills。

## 默认推理后端背景

用户未另行指定、且实际部署证据未表明其他后端时，HBM、权重、KV Cache、激活及运行时占用分析默认采用 **vLLM + vLLM-Ascend（Ascend 执行路径）**，不再重复询问后端名称。用户显式指定的其他后端或已确认的实际部署优先；发生冲突时说明差异，不混用执行路径。

默认后端只确定分析路径，不确定硬件代际、vLLM/vLLM-Ascend 版本、KV dtype、并行布局、chunk、异步、MTP或缓存池配置；这些仍按实际代码、配置和日志取证，关键未知项保留条件并集中补问。优先追踪配套vLLM与vLLM-Ascend的注册、替换及最终生效路径；只找到上游vLLM实现不代表Ascend插件沿用同样布局。模型官方参考实现仅用于解释结构或明确标注的辅助对照，不能替代默认后端的物理容量账，也不能因缺参数而静默切换到参考实现或BF16布局。

## 完整 HBM 估算

当请求是“估算一个模型的 HBM/显存占用”时，使用 [model-hbm-budget](skills/model-hbm-budget/SKILL.md) 总入口，联动下面三个专项 skill。完整方法见 [HBM 占用指南](skills/model-hbm-budget/references/hbm-guide.md)，仓库内自动路由见 [AGENTS.md](AGENTS.md)。总入口只负责编排、统一配置和去重合账，三个专项继续维护各自算法。

| Skill | 用途 |
| --- | --- |
| [kv-cache-budget](skills/kv-cache-budget/SKILL.md) | 根据模型结构、vLLM/vllm-ascend 源码与运行日志，逐组件核算单请求 KV Cache、prefix/chunk、滑窗、压缩器状态与物理池需求；不含多请求或多卡模块。 |
| [weight-hbm-estimation](skills/weight-hbm-estimation/SKILL.md) | 根据模型 checkpoint 与 vLLM/vLLM-Ascend 实际代码，逐模块核算并行切分、共享专家 DP、Engram offload、量化转换及权重副本的 HBM 占用；关键未知配置先询问。 |
| [activation-hbm-estimation](skills/activation-hbm-estimation/SKILL.md) | 根据推理执行源码与运行日志，核算除权重和 KV 外的激活、Attention/Indexer、MoE AllGather/AllToAllV、workspace、HCCL、图池及运行时逐卡峰值；按存储生命周期去重并保留未知项。 |

用户未指定且没有实际负载取值时，默认按序列参考长度 **128K=131072 tokens**、连续前缀命中率 **90%** 估算，明确标注默认假设；实际取值或用户要求优先。命中前缀按后端粒度对齐，命中率不直接乘减整池或整模型显存。

每个模块的中间计算结果均用表格输出，包含参数、公式与数值代入、bytes/GiB、作用域及状态。

默认输出顺序为：**参数与证据 → 各模块计算过程 → 异步调度增量 → 逐阶段合账与容量检查 → 末尾总体汇总表**。末尾汇总表分别列出 KVCache、权重、激活/workspace、图模式/图池、HCCL buffer、其他持久缓冲/通信池、运行时与分配器额外量，以及总计和条件余量。未知项保留符号，不填0；完整格式见[报告模板](skills/model-hbm-budget/references/report-template.md)。

调用流程为：一次收集共享配置 → 三个专项形成分账 → 根据非KV预算求实际KV池 → 计算异步增量 → 同rank同阶段去重合账 → 末尾汇总。读取skill不等于已经完成相应模块的计算。

例如：“使用 $model-hbm-budget 估算模型 X 在这份部署命令下的 HBM，先展开公式、数值代入、异步在途和跨批占用及加载/运行峰值，最后按 KVCache、权重、激活、图模式、HCCL buffer 等给单卡总体汇总表，写明模块名称和配置，不使用上下文指代。”

在其他工作区使用时，将 `skills/` 下四个目录（总入口与三个专项，含 references/scripts）一起安装到 Agent 的 skills 目录；Codex 通常为 `~/.codex/skills/`。安装后可显式调用 `$model-hbm-budget`，也允许按描述自动选择；自动发现取决于宿主的 skill 加载机制。仅更新 GitHub 不会自动安装到本机或其他会话。只需单项时直接调用对应专项。
