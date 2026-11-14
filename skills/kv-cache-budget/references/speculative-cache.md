# 投机缓存：默认纳入，按实际实现分账

## 默认范围与证据不足

用户约定容量预算默认包含投机解码的缓存，而不是将任何模型都假定成同一种draft。优先使用最终speculative_config、draft config、加载日志和实现代码。用户仅给模型名时，可核对原生MTP/nextn与后端支持后提出含原生MTP的明确假设；不能无依据选择任意独立draft模型、推测长度或接受率。缺少会改变结果的关键配置时集中补问，并先交付主模型已知小计加待确定的投机项，不给一个看似完整的精确总量。

确认关闭的实际部署按关闭报告；若用户需要默认含投机预算，另列启用后的条件场景，不将其伪装成当前实测配置。config中num_nextn_predict_layers/mtp_num_hidden_layers等只说明模块结构，不等于启用标志或num_speculative_tokens。不能仅凭未写启动参数判定有无投机。

## 需要确认的参数与调用链

查method、draft模型标识及revision、target/draft各自config、推测token数K、树/多分支形状（若使用）、实际最大验证位置数与lookahead、接受/拒绝后的保留和回滚、P/D角色、TP/CP/PP/DP与层归属。接受率只影响平均有效推进，不能直接缩减峰值已分配页。短于推测窗口的上下文余量须按实际截断逻辑处理。

从模型注册与构造追到proposer/draft加载、speculative_config规范化、get_kv_cache_spec、cache绑定/共享、scheduler.allocate_slots及num_lookahead_tokens、状态复制与释放。静态符号随版本变化，按功能搜索而非固定文件路径：

```bash
rg -n 'speculative_config|num_speculative_tokens|draft_model_config|mtp_num_hidden_layers|num_nextn_predict_layers' <config-model-launch-paths>
rg -n 'get_kv_cache_spec|kv_sharing|share.*cache|draft.*cache|hidden_state.*cache|num_speculative_blocks' <proposer-runner-model-paths>
rg -n 'num_lookahead_tokens|allocate_slots|accepted_tokens|rollback|defer_block_free' <scheduler-manager-paths>
```

## 缓存清单

| 项 | 核算要点 |
| --- | --- |
| 原生MTP/DSpark/EAGLE等附加模块 | 按真实模块owner的Attention/Indexer/卷积/递推state逐项算；不是自动增加一个完整主模型，也不是只算K个draft token |
| 独立draft模型 | 独立读取其层数、KV形状、dtype、窗口与并行配置；全历史draft若需要长上下文，其KV按真实已喂入历史计算，不限于当前K个候选 |
| 主模型验证KV | 按验证阶段真实位置、lookahead、分支及额外尾部保留算页差；若已经包含在H/F/allocate_slots中，不再额外加K |
| 递推/卷积/固定state | 按实际num_speculative_blocks、conv历史扩展、多版本、rollback/checkpoint与恢复副本；不能假定每层固定一份，也不能统一乘K或异步批数 |
| 持久hidden-state cache | 确认是否登记为缓存页或独立持久tensor，按实际shape/dtype/lifetime另列；临时激活与workspace不混成KV |
| 无独立KV的提议方式 | 代码证明ngram/lookup等不建draft KV时，draft KV可注明不适用/0；仍核算target验证附加页，不由“投机”名称虚构整份模型缓存 |

跨模块共享权重、共享lm_head或复用TopK不证明KV也共享。只有实际绑定到同一存储/owner的KV才去重；层别名和共享backing views不重复算。独立draft的prefix命中能力和可复用页另查，不能自动沿用target的90%命中率。

## 与prefix/chunk、异步和物理池合并

对target和每个实际draft/MTP owner都给出prefix、已结算Q、历史窗口/固定state以及各批在途记录。保持用户要求的输出顺序：先合并去重后的已结算基线（其中包含draft历史和固定槽），最后统一加入target验证与draft候选的异步在途新增页，再补额外保留。

```text
完整预算 = target与draft去重后的已结算基线
         + target验证/draft提议的在途新增页
         + 尚未计入的checkpoint、回滚、connector/fence保留
```

target与draft可能有不同执行次数和位置集合，需按该版本调度逻辑分别验证公共额度；不能机械使用async_batches*K*batch_budget，不能给每个请求重复分配公共token上限。模型state、额外lookahead若在规格/allocator里已计入，再次加算会重复。

MTP/draft可能改变page buckets、组数、tuple slots、block size及PP放置，应使用包含投机模块的最终spec集合重做分组，不能在旧主模型统一pool成本上随意加一个原始payload。独立池/独立设备则分别计费，不强行合用一个U。P未执行draft不代表D无需draft KV；是否在P加载、物化、传输及D是否重新prefill由实际流程决定，不强行给P/D各加同一份。

## 输出门禁

结果至少写明“含投机/不含投机/仅主模型已知小计”，列method、draft标识/附加owner、K、target验证附加页、state槽、prefix共享与P/D放置。待确认项写成条件公式或单列未知，不默认为0。纯主模型旧估算不能直接改标签为含投机，需要重新核对布局和调度。投机模型权重、临时hidden activations、logits与workspace属于非KV预算，避免遗漏整机HBM时再重复扣除。
