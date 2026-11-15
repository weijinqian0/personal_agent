# 单请求 prefix / chunk / 状态生命周期

## 快照与归因

`P` 为有效命中前缀，`Q` 为此后安全结算历史，`S=P+Q`；有序未结算 chunk 为 `C_i`，`F=sum(C_i)`，`H=S+F`。已结算与在途位置必须分列；L-P 不等于某个 chunk。

长历史组件按累计记录函数 R(x) 差分。普通 KV 为 R(x)=x；压缩边界依真实实现，常见为 floor(x/r)。prefix、Q 和各 C_i 的记录和应为 R(H)，页数则按对应位置的物理页并集计算，不能逐列 ceil 后相加。

## 每个 state 先查五件事

| 生命周期 | 要查的实际行为 |
|---|---|
| 写入 | 哪个 kernel 在哪些输入位置写入，是否产生压缩记录或更新固定槽 |
| 恢复 | prefix 命中时恢复哪些 checkpoint，复用、复制还是重新计算 |
| 保留 | 尾窗、checkpoint、推测回滚版本分别留多久 |
| 释放 | 安全结算、窗口滑过、回滚完成时何时回收；是否有在途保护 |
| 去重 | 恢复态、执行态、checkpoint 是否指向相同 backing |

名字或 tensor shape 不能代替生命周期证据。固定 state 已分配时，新 chunk 可能没有新增字节，但原有容量仍需计入。

## 窗口与 checkpoint

SWA 和窗口 compressor state 分列：命中边界所需尾窗、实际保留的 prefix checkpoint、活跃历史尾窗、每批在途新增位置。窗口 W 既不是整个 prefix 容量，也不证明只存一个窗口。

```text
resident_pages = prefix_checkpoint_pages ∪ active_pages ∪ extra_retained_pages
extra_prefix_pages = prefix_checkpoint_pages - active_pages
```

按物理页身份去重。未知 checkpoint 保留策略时写集合公式与缺项，不把未知当零。主 KV 与 Index state 各用自己的 dtype、窗口和页规则。

## 异步增量

固定安全结算位置 S，分别计算基线和含所有 C_i 的快照：

```text
增量 = bytes(实际保留页集合，含全部在途位置) - bytes(安全结算基线页集合)
```

覆盖所有会写入的组件，不能只扩大 SWA。固定槽仅在真实增加执行版本时增加；压缩记录逐批做累计差分。不默认在途批次数或 chunk 长度，配置上限与实际写入量分开说明。

最终按“prefix → Q → 活跃窗口/固定 state → 在途新增页 → 未计入的额外保留”汇总。若页已在固定缓存池内，增量只是消耗池内空闲空间，不再加到 allocated_pool_bytes。
