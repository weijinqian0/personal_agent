# 从资料到容量账

先定位用户实际 checkout，记录 `git rev-parse HEAD`、相关 dirty diff、模型 config 和最终解析的运行配置。路径是搜索入口，符号和文件会随版本移动；不导入生产模型、不下载权重也能完成静态追踪。

## 1. 运行条件和模型结构

```bash
rg -n 'max_model_len|max_num_batched_tokens|max_num_seqs|block_size|cache_dtype|async_scheduling|speculative|kv_transfer' <log-or-launch-directory>
rg -n 'num_hidden_layers|num_key_value_heads|head_dim|compress_ratios|sliding_window' <model-config>
```

查模型构造中 Compressor、Indexer、SWA、MTP/DSpark 的条件分支。config 给能力，实际构造和启动参数决定激活的路径。论文帮助解释结构；实际字节布局取后端。

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
```

常见文件：`patch/platform/patch_kv_cache_utils.py`、上游 `config/vllm.py`、`v1/kv_cache_interface.py`、`v1/core/sched/scheduler.py`、`v1/core/kv_cache_manager.py`、`single_type_kv_cache_manager.py`。跨版本确认方法签名与调用点。

分组填充可能是优化器选出的 approximate GCD，不一定等于最大专家/Attention 层计数。计算物理 backing 时注意不同 group 可复用对应槽，但分配器 block ID 是否独立仍要确认。

## 4. 前缀、并行与连接器

```bash
rg -n 'prefix_cache|retention_interval|ref_cnt|copy_on_write' <cache-manager-directory>
rg -n 'state|cache_families|kv_role|compressor' <kv-transfer-directory>
```

确认 checkpoint 保存哪些页、命中粒度、外部缓存恢复哪些状态、producer 何时释放、consumer 何时获得所有权。物理前缀共享不代表访问流量同比例共享。TP 按真实 KV 头身份、CP 按上下文归属、PP 按层归属，DP 按请求归属计量；EP 不自动切 KV。

## 5. 日志核对

```bash
rg -n 'registered_bytes|num_blocks|cache group|token capacity|Maximum concurrency' <logs>
```

记录 `pool bytes/block`、有效请求持块、总池预分配三个口径。容量日志可能使用 max_model_len 和 max_in_flight_tokens 的 admission 成本折算，不能用 token capacity 直接反推每 token 长期 KV 大小。验证计算只证明日志算术一致；本地 commit 未匹配运行环境时保留该证据缺口。
