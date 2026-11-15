# MTP / DSpark 附加缓存

保留实际投机模块的缓存核算，不强制任意 draft 模型。查最终 speculative_config、模块 config、加载与 proposer 路径；确认关闭则说明，未知列补充项。

附加模块层数与推测 token 数 K 是不同参数。接受率不能直接缩减峰值页数。

| 组件 | 核算方式 |
|---|---|
| MTP/DSpark 附加 Attention / Index / state | 按真实 owner、shape、dtype、窗口、固定槽逐项算；不能自动加整份主模型 |
| 独立 draft | 用其自身历史长度与组件结构，不限于 K 个候选 token，也不自动沿用 target 命中率 |
| Target 验证 KV | 真实验证位置、lookahead 和分支带来的新增页；已包含在 H/F 的位置不再加 K |
| 递推/卷积/回滚 state | 按实际槽位、执行版本、checkpoint 与恢复副本，不统一乘 K 或 chunk 数 |
| 持久 hidden-state cache | 按真实 shape/dtype/lifetime 列项；临时激活和 workspace 不算 KV |

对 target/draft 都保留 prefix、Q、各 C_i 与窗口/state 分账。共享权重不证明共享 KV；仅相同物理 backing 去重。包含附加模块后重新确认 allocator 分组、尾槽与 padding，不能只在旧池成本上加原始 payload。

```bash
rg -n 'speculative_config|num_speculative_tokens|draft_model_config|num_nextn_predict_layers' <config-model-paths>
rg -n 'get_kv_cache_spec|share.*cache|hidden_state.*cache|num_speculative_blocks' <proposer-runner-paths>
rg -n 'num_lookahead_tokens|allocate_slots|accepted_tokens|rollback' <scheduler-manager-paths>
```

结果注明含投机、不含投机或仅主模型已知小计，并列出方法、附加 owner、K、验证页、state 槽及未知项。
