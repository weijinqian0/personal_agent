# 已知几何的预算复算

部署总量的完成状态与allocator证据按[物理布局核对](physical-layout-validation.md)判定。`calculate_pool.py`仅复算输入声明的U和块数，不验证物理映射；可另用`audit_layout.py`核对声明几何与成本冲突，不能将脚本成功退出视为运行部署已验证。

本例来自一次用户提供的 V4-Flash 部署分析：base block=32，日志给出每池 block 592000 B，含 DSpark 的分组为 C4、C128、两组 SWA、C4 state、C128 state。它是已给定几何的回归算例，不是任意 A5/V4 的默认布局；原文也未证明本地快照与远端完整 commit 一致。

该样例主记录 608 B、索引 132 B，页大小分别 19456/4224 B，25 slots，故 `25*(19456+4224)=592000`。不能与其他版本的 640 B 主记录、22 slots、base block=128 混用。

脚本输入 JSON：

```json
{
  "pool_bytes_per_block": 592000,
  "max_model_len": 1048576,
  "global_in_flight_limit": 32768,
  "groups": [
    {"name":"c4", "kind":"full", "block_tokens":128},
    {"name":"c128", "kind":"full", "block_tokens":4096},
    {"name":"swa", "kind":"window", "block_tokens":32, "window":128,
     "multiplicity":2, "boundary_slack_blocks":1},
    {"name":"c4_state", "kind":"window", "block_tokens":2, "window":8,
     "boundary_slack_blocks":1},
    {"name":"c128_state", "kind":"window", "block_tokens":4, "window":128,
     "boundary_slack_blocks":1}
  ],
  "requests": [{"id":"A", "history_tokens":131072, "in_flight_tokens":8192}]
}
```

`history_tokens` 是该预算快照采用的完整长度（含本轮预计写入），不是前缀长度加 chunk 后再重复相加。`in_flight_tokens` 是该请求的实际值或明确的场景上界。`max_model_len` 是窗口截断上界。窗口可选 `extra_retained_tokens`；固定分配组支持 `kind: fixed, blocks: N`。

`multiplicity` 表示相同规则的独立分配组数量，不是层数；层/槽位和页填充已包含在 pool 单位成本内。仅适用于独立 group block 消耗可加的共享池：有跨请求共享物理页、prefix 检查点或 connector 持块时，需在外部去重/补充，脚本输出是未去重的条件预算。

该脚本不输出逐组件prefix/chunk分账，也不推导checkpoint retention、PD延迟释放或layerwise物理buffer reuse。正文须另外按[分账与释放生命周期](prefix-chunk-lifecycle.md)列明这些项；只给本脚本的总数，不能称为包含全部prefix cache的节点驻留量。

| 请求长度 | 请求 F | 总块数 | bytes | GiB |
| ---: | ---: | ---: | ---: | ---: |
| 131072 | 8192 | 7760 | 4593920000 | 4.278421 |
| 131072 | 16384 | 14416 | 8534272000 | 7.948160 |
| 131072 | 32768 | 27728 | 16414976000 | 15.287638 |

两请求各 F=8192 时合计 9187840000 B；各 F=16384 时合计 17068544000 B。两者分别使用公共 16K/32K 在途预算，不能把32K给每条请求后还称公共32K。

容量日志回归：T=1048576，F=32768 时总块数35120；池43869块，则 `floor(43869/35120*1048576)=1309794`。这是 admission 折算，与长期 KV token capacity 不同。

若单请求128K且实际连续命中90%，剩余约13108 token。它不能仅因配置支持32K在途就拥有32K新增prefill；本例论文/日志的本地 prefix caching 关闭，外部池命中仍须单独取证。8K/16K调度预算共享，多请求同时驻留也不会自动使算力吞吐按并发线性增长。
