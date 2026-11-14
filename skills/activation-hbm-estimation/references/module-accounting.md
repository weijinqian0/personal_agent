# 模块与源码追踪

按符号定位运行版本，下面是入口线索，不是跨版本固定路径。

## 基础尺寸

T_r是实际输入token行数（含padding），H是hidden size，I_r是本卡专家中间维，N_recv是本卡接收的token-expert对数。

| 存储 | 常见逻辑尺寸 | 必查 |
|---|---|---|
| hidden/residual | T×H×元素bytes | replicated/SP；切片是否持有完整backing |
| router logits | T×E×router元素bytes | FP32、gather和辅助score |
| top-k | T×k×(权重bytes+索引bytes) | cast、mask、expanded/reversed mapping副本 |
| 专家输入 | N_alloc×H×通信元素bytes | N_alloc未必是N_recv；capacity/active_num |
| gate/up | N_alloc×2I_r×输出bytes | 融合是否消除物化；INT32/BF16 |
| SwiGLU | N_alloc×I_r×输出bytes | 量化前后是否共存 |
| 专家输出 | N_out_alloc×H×输出bytes | 沿用输入行数还是有效group计数 |
| logits | T_logits×V_r×输出bytes | 全词表gather、FP32、仅采样位置 |

这些是尺寸分项，不是可直接求和的峰值公式。

## AllGather / AllToAllV

入口符号：`PrepareAndFinalizeWithAllGather/All2All`、`TokenDispatcherWithAllGather/All2AllV`、`apply_moe_mlp`、`apply_gmm1_act_quant/apply_gmm2`。

- AG：输入/SP分片 → 可选先量化 → TP/DP/EP gather → top-k → 本地专家routing → GMM1/activation/quant/GMM2 → unpermute → reduce/scatter。
- A2AV：输入/SP分片 → 本地top-k展开排序 → 可选量化 → 发/收 → 专家重排 → GMM → 逆重排 → 反向发/收 → 本地聚合/恢复。

发送对数为`T_source_local×k`，均衡接收为`T_unique×k/EP`。两者相等不意味着buffer可以共用。

查Python empty/cat/contiguous、op-plugin apply_tensor_without_format、C++ at::empty和workspace tiling。有些AG路径以全局`active_num=T_global×k`申请路由输入，active_expert_range只筛计算；GMM若沿用x.size(0)，激活/输出继续按全容量分配。其他版本可能compact，不泛化。

追上层`mlp_output`：逆重排返回新tensor时，原输出可能被外层引用保留至反向通信结束，形成原输出、发送、接收三份。resize_(0)/dispose需核对别名和同步。

完整EP专家的I_r不再除Attention TP；只有真实专家内部TP才缩小I_r。EPLB冗余、capacity、倾斜和共享专家改变分配/共存项。

## Attention / Indexer

核对backend、头分片、DSA-CP token分片、PCP/DCP、MLA吸收。吸收式MLA可能出现：

- Q低秩投影及norm：`[T_r,q_lora_rank]`。
- 普通/旋转Query：`[T_r,n_heads_local,d_nope/d_rope]`。
- 吸收Query：`[T_r,n_heads_local,kv_lora_rank]`。
- cat后的新Query：`[T_r,n_heads_local,kv_lora_rank+d_rope]`。
- latent输出：`[T_r,n_heads_local,kv_lora_rank]`。
- V投影输出：`[T_r,n_heads_local,v_head_dim]`。
- Indexer query/scale、`[T_r,n_index_kv_heads,index_topk]`索引、共享top-k buffer。

view与backing去重；RoPE/cat新存储另计。cat函数返回后可释放，所以latent、V投影和cat不一定同一时刻共存。

调用方可能提前分配完整`[T_dp,H]`输出；内部只处理T_dp/TP行，仍可能保留完整输入backing和输出。Layer0完整residual与后续SP residual不同，逐阶段查。

临时KV投影/gather/packed/反量化计入，主KV/Indexer KV常驻池排除。共享top-k属于非KV缓冲，按owner计一次，不乘层数。

融合Attention不默认物化完整Q×上下文score；tile score、离散KV gather、split-KV partial、softmax统计仍占HBM。沿GetWorkspaceSize、平台核数、tile和系统workspace取证，不确定时保留W_attn/W_indexer。

## 常见源码入口

- `attention/sfa_v1.py`、`attention/context_parallel/sfa_cp.py`：GLM MLA/DSA-CP。
- `attention/indexer.py`、`device/device_op.py`：索引、Query拼接和设备分支。
- `ops/mla.py`、上游模型：外层输出/持久buffer。
- `ops/fused_moe/{prepare_finalize,token_dispatcher,moe_comm_method,moe_mlp}.py`。
- `quantization/methods/`：量化注册、融合默认、scale和dispose。
- `csrc/gmm/`、`csrc/attention/`：adapter和workspace。

这些相对路径通常位于vllm_ascend下，csrc例外。官方[op-plugin](https://github.com/Ascend/op-plugin)也要查实际配套版本，不能将master当目标镜像。
