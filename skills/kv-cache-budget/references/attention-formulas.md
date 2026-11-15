# Attention 状态公式

## 统一符号与计算范围

先区分Attention结构与缓存组件：GQA/MLA可以同时带有SWA、Indexer或compressor state。模型包含几种实际存储就列几行，不把这些名称当互斥类别。

以下按单请求、明确计算范围内的实际shape计算：P为命中prefix；Q为之后已安全结算历史；S=P+Q；C_i为各未结算批次，F=sum(C_i)，H=S+F。L为实际存储owner数，不自动等于模型层数；n_q为Query头数，n_kv为实际存储KV头数，d为维度，b为元素字节，s为一条记录的存储字节。层间参数不同则逐owner求和。

R(x)表示前x个原始位置累计产生的缓存记录数。令Delta_R=R(S+F)-R(S)，对没有token压缩的组件R(x)=x。表内长历史组件prefix列计算本请求命中的P。窗口与state的prefix列需要真实checkpoint集合，不能只由P确定。

## Prefix cache与chunk双列公式表

这些是组件自身的记录/状态字节，不含统一pool额外padding和临时workspace。固定执行state不属于“每chunk新产生的token KV”，必须明确列出而非硬乘F。

| Attention结构／缓存组件 | Prefix cache占用 | Chunk执行占用，含全部未结算批次 |
| --- | --- | --- |
| MHA，等宽等精度K/V | `L*P*2*n_q*d*b`，实际n_kv=n_q | `L*F*2*n_q*d*b` |
| GQA，等宽等精度K/V | `L*P*2*n_kv*d*b` | `L*F*2*n_kv*d*b`，不能用Query头数替代KV头数 |
| MQA，等宽等精度K/V | `L*P*2*d*b`，n_kv=1 | `L*F*2*d*b` |
| K/V不等宽或不同精度 | `L*P*(n_k*d_k*b_k+n_v*d_v*b_v)` | `L*F*(n_k*d_k*b_k+n_v*d_v*b_v)` |
| 紧凑MLA | `L*P*s_MLA`；若独立存latent和RoPE，`s_MLA=d_c*b_c+d_r*b_r` | `L*F*s_MLA`；latent兼供K/V不乘2，RoPE已在记录内不再加一次 |
| 展开MLA | `L*P*s_expanded`，s按实际展开K/V与独立分支shape计算 | `L*F*s_expanded`，不能套紧凑latent公式 |
| 压缩MLA／主KV | `L*R(P)*s_main`；常见R(x)=floor(x/r)，实际边界按代码 | `L*Delta_R*s_main`，不能只对F单独除压缩比后取整 |
| Indexer K | `L_index*R_index(P)*s_index`；s包含实际存储K向量与scale等 | `L_index*Delta_R_index*s_index`；不默认有独立V，不默认乘index Query头数 |
| folded／重排索引 | 独立副本按实际记录映射与字节；alias不重复计费 | 各批新增独立存储按实际shape；别名view不制造额外副本 |
| SWA | 各保留checkpoint窗口页的物理并集；既不默认全P也不默认只一个W | 新增行预算`L_swa*F*s_swa`；另保留活跃历史尾窗，分页后与prefix页去重 |
| 滑窗compressor state | 主/索引state各自保留checkpoint页的并集 | 新增行预算`L_state*F*s_state`，`s_state=(D_kv+D_score)*b_state`；另保留自身历史尾窗 |
| 固定ring | 不支持prefix复用则标不适用；支持则按真实checkpoint快照 | 每份`L_owner*ring_rows*state_dim*b_state`，按实际执行槽数/版本数分配，不乘F |
| 递推／线性Attention state | 支持时按唯一checkpoint状态快照；不支持则不适用 | 例如单owner矩阵`n_state_heads*d_k*d_v*b_state`；加实际其他状态，按执行版本数分配，不乘F |
| 卷积state | 支持时按实际保留快照去重 | 执行cache的tensor元素数*element_size，按实现的槽位/版本数计算，不假定随F线性增长 |

长历史组件不能漏掉Q，其独立账为`L*[R(P+Q)-R(P)]*s`。按批展开时用`R(S+sum(C_1...C_i))-R(S+sum(C_1...C_(i-1)))`；各批相加须等于Delta_R，不能在包含F的H上再追加F。

固定state若已经有执行槽，继续一个chunk未必新增分配，但不能把该执行槽容量误写成0；prefix快照恢复是否产生额外副本、异步/推测是否多版本按代码确认。Checkpoint集合与执行槽可能alias或copy-on-write，最终仍按实际物理身份去重。

## 窗口、checkpoint与物理页

SWA和滑窗state都需要“命中边界所需尾窗｜整个prefix池保留页｜活跃历史尾窗｜各批在途”分账。历史尾窗常见为min(S,W-1+extra)，实际取代码。稳态SWA有效量`sum_l(min(H,W_l)*s_l)`只适用于对应稳态口径，不能代替prefill在途或全部prefix池。

对于已确认采用以下规则的管理器：

```text
n_active = ceil(min(max_model_len,W-1+extra+F)/block_tokens) + slack
resident_pages = prefix_pages ∪ active_pages
extra_prefix_pages = prefix_pages - active_pages
```

每个state使用自己的窗口、逻辑block、dtype和回收规则，不能按主KV压缩比把state也缩小。命中所需尾窗按匹配逻辑计算，不能直接把admission的slack追加到每个checkpoint。保留边界、页重合与淘汰按真实策略；详细流程见[prefix/chunk生命周期](prefix-chunk-lifecycle.md)。

普通独立分页按唯一页数*实际页字节求和；统一pool仅在布局已确认时按U*独立group唯一block数计算。同组主KV、索引和folded不能各乘一次完整U；不把组件字节表直接当物理池预算。

压缩器缓冲与长期压缩记录分别计。压缩比只确定记录产出频率，不一定等于输入感受野（例如相邻块重叠）。state 可能保存 kv/score 投影行、固定递推矩阵或在线聚合统计量，必须读实现，名称不足以确定布局。

量化记录字节按实际打包、分组尾部、scale、zero-point 和 padding 求和。`values*(bits/8 + scale_bytes/group_size)` 仅用于完整分组且无其他开销的简化情形。

全局扫描类 Attention 的 decode 逻辑旧历史读可近似等于所访问历史的有效容量；稀疏 Attention 另列全局/候选索引扫描、选中主记录和局部窗口，不能以驻留容量替代访问量。物理 HBM 重读取决于 kernel，不能把逻辑唯一载荷称为实测带宽流量。
