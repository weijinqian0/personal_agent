# 从资料到容量账

先定位用户实际 checkout，记录 `git rev-parse HEAD`、相关 dirty diff、模型 config 和最终解析的运行配置。路径是搜索入口，符号和文件会随版本移动；不导入生产模型、不下载权重也能完成静态追踪。

## 1. 运行条件和模型结构

```bash
rg -n 'max_model_len|max_num_batched_tokens|max_num_seqs|block_size|cache_dtype|async_scheduling|speculative|kv_transfer' <log-or-launch-directory>
rg -n 'chunked_prefill|hybrid_kv_cache|attention_backend|num_speculative_tokens|lookahead|extra_retained_tokens|prefix_cache|retention_interval|preemption|gpu_memory_utilization|kv_cache_memory_bytes|num_gpu_blocks_override' <log-launch-and-config-directories>
rg -n 'num_hidden_layers|num_key_value_heads|head_dim|compress_ratios|sliding_window' <model-config>
```

查模型构造中 Compressor、Indexer、SWA、MTP/DSpark 的条件分支。config 给能力，实际构造和启动参数决定激活的路径。论文帮助解释结构；实际字节布局取后端。

参数名称和优先级随版本变化；若未命中，查对应CLI注册与配置dataclass。追踪最终解析值和设备能力分支；区分“未填写”“默认启用”“实际不支持”。多模态长度按encoder输出及送入语言主干的位置数取证，不能只按文本tokenizer计数。

block_size沿“启动/最终解析配置 → vllm-ascend平台与模型校验/覆盖 → Attention backend支持块大小 → 配套vLLM配置默认”取证，记录实际覆盖顺序。搜索`block_size is None`、`check_and_update_config`、`get_supported_kernel_block_sizes`及模型block映射；符号随版本变化。找不到最终值时使用SKILL.md中用户约定的128回退，不把它标为代码默认。单独记录storage_block_size、压缩比和ring capacity，不能互相替代。

## 2. 缓存登记与页字节

```bash
rg -n 'def get_kv_cache_spec|state_dim|state_dtype' <backend-model-directory>
rg -n 'real_page_size_bytes|unpadded_page_size_bytes|page_size_padded|storage_block_size|tokens_per_state' <cache-interface-files>
rg -n '_allocate_kv_cache_tensors|_reshape_kv_cache_tensors|block_stride|layer_stride' <runner-directory>
```

vllm-ascend 的 V4 常见入口：

- `vllm_ascend/models/deepseek_v4/model.py`：Attention 组成、SWA Spec。
- `vllm_ascend/models/layer/attention/layer.py`：压缩主 KV Spec、设备相关块表。
- `vllm_ascend/models/deepseek_v4/indexer.py`：Index K dtype/scale 和索引压缩器。
- `vllm_ascend/models/deepseek_v4/compressor.py`：FP32 state、宽度、padding、算子参数。
- 上游 `vllm/models/deepseek_v4/compressor.py`：父类窗口、注册与绑定。
- `vllm_ascend/core/kv_cache_interface.py`：页字节与自定义 Spec。

查看继承链、dtype resolver 和设备分支；不能将 `head_size` 数字直接解释成字节。若分析 state 更新语义，继续查对应算子的 state 读写、slot mapping、block table，区分分配预算和实际写入行为。

## 3. 分组、调度与回收

```bash
rg -n 'group_and_unify|num_layer_tuples|pool_bytes_per_block|max_memory_usage_bytes_from_groups' <cache-planner>
rg -n 'max_concurrent_batches|max_in_flight_tokens' <vllm-config-directory>
rg -n 'token_budget|num_in_flight_tokens|remove_skipped_blocks|max_admission_blocks_per_request' <scheduler-and-manager-directory>
rg -n 'async_scheduling is None|max_concurrent_batches|max_in_flight_tokens|processed_step_seq|deferred_frees' <config-and-scheduler-directories>
```

常见文件：`patch/platform/patch_kv_cache_utils.py`、上游 `config/vllm.py`、`v1/kv_cache_interface.py`、`v1/core/sched/scheduler.py`、`v1/core/kv_cache_manager.py`、`single_type_kv_cache_manager.py`。跨版本确认方法签名与调用点。

分组填充可能是优化器选出的 approximate GCD，不一定等于最大专家/Attention 层计数。计算物理 backing 时注意不同 group 可复用对应槽，但分配器 block ID 是否独立仍要确认。

## 4. 前缀、并行与连接器

```bash
rg -n 'prefix_cache|retention_interval|ref_cnt|copy_on_write' <cache-manager-directory>
rg -n 'reachable_block_mask|replay_boundaries|shared_prefix_boundary|find_longest_cache_hit|free_blocks' <cache-manager-directory>
rg -n 'state|cache_families|kv_role|compressor' <kv-transfer-directory>
rg -n 'request_finished_all_groups|request_finished|delay_free_blocks|defer_block_free|finished_sending|finished_recving' <connector-and-scheduler-directories>
rg -n 'use_layerwise|wait_for_layer_reuse|apply_layerwise_kv_cache_plan|supports_layerwise_buffer_reuse' <backend-directory>
rg -n 'attention_tp|dp_attention|tensor_parallel|data_parallel|pipeline_parallel|prefill_context_parallel|decode_context_parallel|pcp_size|dcp_size' <launch-config-and-model-directories>
```

确认 checkpoint 保存哪些页、命中粒度、外部缓存恢复哪些状态、producer 何时释放、consumer 何时获得所有权。物理前缀共享不代表访问流量同比例共享。TP 按真实 KV 头身份、CP 按上下文归属、PP 按层归属，DP 按请求归属计量；EP 不自动切 KV。

prefix命中应从实际boundary匹配逻辑推导最少窗口页，从retention mask和淘汰逻辑推导全部驻留页。检查被请求释放的页是否仍带hash进入可复用free queue，不能把“释放引用”解释成立即清空缓存，也不能把可淘汰缓存全当成不可用容量。PD沿connector返回值追到scheduler的OR合并、fence与最终free，不依据layerwise名称猜测释放结果。具体分账方法见[prefix/chunk生命周期](prefix-chunk-lifecycle.md)。

特别检查backend patch是否绕过上游config字段、读取旧环境变量或重写协调器构造；只有沿运行版本的调用链追到实际retention成员和cache_blocks入参，才能确认生效默认。配置字段、文档与实际分支冲突时显式报告差异，不照搬字段默认值，也不未经授权修复生产代码。压缩混合模型的命中粒度需查各group的逻辑block及协调器LCM/partial-hit逻辑，不直接用base block对齐名义命中率。

对 P、D 独立追踪 Attention 的实际 process group，以及每个组件最终的 local shape。启动参数 TP、EP 不能替代模块内部的 Attention TP。MLA latent、索引或 compressor state 可能与 Query heads 使用不同的复制/切分规则；CP 也可能使窗口重叠、边界页或 state 在多个 rank 保留。重新生成各 stage/rank 的 cache groups 和 pool bytes/block，不能先算完整模型池填充再机械除卡数。

若最终配置/日志未给足布局，向用户说明缺的是哪项、会影响哪个组件，索取完整启动命令或必要参数。输出中标注已确认/假设/待补充；不把未知并行值悄悄置为1。

## 5. 日志核对

```bash
rg -n 'registered_bytes|num_blocks|cache group|token capacity|Maximum concurrency' <logs>
```

记录 `pool bytes/block`、有效请求持块、总池预分配三个口径。容量日志可能使用 max_model_len 和 max_in_flight_tokens 的 admission 成本折算，不能用 token capacity 直接反推每 token 长期 KV 大小。验证计算只证明日志算术一致；本地 commit 未匹配运行环境时保留该证据缺口。

计算可接纳请求数前，查实际池大小是否受显式bytes预算或块数override控制，核对它们相对内存利用率的优先级。对照weight/non-framework/activation/graph/workspace等profiling项，确认哪些已从可用池扣除；不重复扣减，也不把这些非KV内存混入KV有效字节。需要实际瞬时峰值时采集或索取逐rank的活跃/空闲/保留块与在途队列数据；仅有启动容量日志不能证明运行峰值。
