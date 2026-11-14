# Attention 状态公式

单请求长度 H，层数 L，KV 头数 n_kv，Query 头数 n_q，头维 d_h，元素字节 b。不同层形状不同时逐层求和。以下为有效内容，不含分页、复制和临时工作区。

| 组件 | 长期容量 |
| --- | --- |
| MHA/GQA/MQA，等宽 K/V | `H * sum_l(2*n_kv_l*d_h_l*b_l)`；MHA 的 n_kv=n_q，MQA 的 n_kv=1 |
| K/V 不等宽或不同精度 | `H * sum_l(n_k_l*d_k_l*b_k_l + n_v_l*d_v_l*b_v_l)` |
| 紧凑 MLA | `H * sum_l(d_c_l*b_c_l + d_r_l*b_r_l)`；latent 兼供 K/V 时不乘 2 |
| 展开 MLA | 按实际展开头数、K/V 各宽度和额外分支计；不能使用紧凑公式 |
| SWA | `sum_l(min(H,W_l) * record_bytes_l)`；执行期另计在途页 |
| 压缩主 KV/索引 | `sum_l(floor(H/r_l) * (main_record_bytes_l + index_record_bytes_l))`，边界按实现 |
| 固定递推矩阵 | `sum_l(state_heads_l*d_k_l*d_v_l*b_state_l)`，不随 H 增长 |
| 短卷积 | 按实际 cache tensor shape × element_size，不假定槽数必为 kernel_size-1 |
| 混合结构 | 逐组件相加；owner 缓存按 owner 数，consumer 不额外制造持久副本 |

压缩器缓冲与长期压缩记录分别计。压缩比只确定记录产出频率，不一定等于输入感受野（例如相邻块重叠）。state 可能保存 kv/score 投影行、固定递推矩阵或在线聚合统计量，必须读实现，名称不足以确定布局。

量化记录字节按实际打包、分组尾部、scale、zero-point 和 padding 求和。`values*(bits/8 + scale_bytes/group_size)` 仅用于完整分组且无其他开销的简化情形。

全局扫描类 Attention 的 decode 逻辑旧历史读可近似等于所访问历史的有效容量；稀疏 Attention 另列全局/候选索引扫描、选中主记录和局部窗口，不能以驻留容量替代访问量。物理 HBM 重读取决于 kernel，不能把逻辑唯一载荷称为实测带宽流量。

对等宽全注意力，已有 S 个位置、本轮 P 个 query：

```text
pairs = B*(P*S + P*(P+1)/2)
QK+PV effective FLOPs = 4*n_q*d_h*pairs   # 再按层求和
```

GQA 共享 KV 不减少 Query 头数；完整矩形、有效因果和后端 padding FLOPs 不混计。FlashAttention 改变执行和中间分数物化，不自动改变持久 KV 结构。相同总 query token、不同请求上下文长度也可能有不同 Attention FLOPs。
