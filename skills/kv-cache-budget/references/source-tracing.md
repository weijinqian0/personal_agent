# 核心源码追踪

先记录模型 config、后端固定 revision、实际启动配置与相关修改；按当前代码搜索符号，不把文件路径当跨版本事实。默认 vLLM + vLLM-Ascend 时必须追到插件实际生效路径。

| 路径 | 需要得到的证据 |
|---|---|
| 模型注册与构造 | 实际 Attention 类型、层/owner 数、Index、SWA、主/索引压缩器、MTP/DSpark |
| cache spec | get_kv_cache_spec、最终 dtype/scale、压缩记录、window、logical/storage block |
| 分配与绑定 | runner allocation/reshape、group/bucket/tuple、padding、alias、固定槽、真实 backing 字节 |
| 写入与恢复 | kernel 输入输出、prefix checkpoint 恢复、复制/复用、state 版本 |
| 调度与释放 | 每个 chunk 的位置、实际异步开关、lookahead、窗口淘汰、回滚与延迟回收 |
| 日志 | 同一快照与缓存池的 group geometry、块数、唯一 registered bytes |

```bash
rg -n 'get_kv_cache_spec|cache_dtype|page_size_bytes|block_size|kv_cache_groups' <backend-paths>
rg -n 'compress|indexer|sliding_window|state|checkpoint|restore' <model-kernel-paths>
rg -n 'allocate_slots|lookahead|async_scheduling|free_blocks|rollback' <runner-manager-paths>
rg -n 'mtp|dspark|speculative_config|num_speculative_tokens' <model-proposer-paths>
```

对每个 state 保留 write/restore/retain/release/deduplicate 的文件位置和解释。只找到 spec 或论文结构不能证明实际分配成本；日志字节/块数比值只有在同一分配范围内才可复用。缺源码或日志时保留条件与未知，不静默切换为参考实现。
