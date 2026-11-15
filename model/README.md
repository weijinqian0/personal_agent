# 模型 HBM 推测档案

保存各模型在特定硬件、并行、量化、负载与后端版本下的 HBM 静态分账、规划假设和可视化。这里是估算数据档案，不是统一的实测基准，也不代表已经验证可部署。

| 模型 | 场景 / 日期 | 内容 | 状态 |
|---|---|---|---|
| [DeepSeek V4 Flash](deepseek-v4-flash/README.md) | A5、TP4/DP4、有效EP16参考、原生量化、FP8 KV、8K/DP；2026-11-15 | [上下文与结果](deepseek-v4-flash/2026-11-15/README.md)、[结构化数据](deepseek-v4-flash/2026-11-15/estimates.json)、[HTML图表](deepseek-v4-flash/2026-11-15/report.html) | 部分静态估算；KV ×2为用户规划假设；完整峰值未闭合 |

## 存放约定

```text
model/
  <model-slug>/
    README.md                 # 模型档案索引
    <snapshot-date>/
      README.md               # 上下文、参数、推导、结果和未决项
      estimates.json          # bytes为主；未知值保留null
      calculate.py            # 复算已归档的几何与算术
      sources.json            # 参考后端revision、源路径、哈希及文档
      report.html             # 可离线打开的说明与图表
```

每份快照必须区分用户确认值、参考场景、规划假设和未知；保留原始并行写法与实际计算解释。GiB=2³⁰ bytes；序列K/M的换算需明确。按KV、权重、激活/workspace、图池、HCCL、持久缓冲及runtime分类，避免把请求页需求替代设备预分配池或重复计算同一storage。

异步持块页并集与手动安全系数分别列示；未覆盖项不能填0。完整峰值或KV物理需求未通过核验时保留null。`calculate.py --check`只证明归档算术一致，不替代源代码/生命周期核验、KV发布门禁或上板实测。新增模型按独立目录归档；不同日期保留历史，不把过期假设覆盖成当前实况。

估算方法仍以仓库的[总入口 skill](../skills/model-hbm-budget/SKILL.md)及三个专项为准。仅推送档案不会自动安装skill或修改推理服务。
