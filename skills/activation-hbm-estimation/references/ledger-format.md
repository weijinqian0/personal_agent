# 单rank storage账本

`scripts/calculate_peak.py --input ledger.json` 使用Python标准库，无模型执行或联网。每份输入代表一个rank和一组同时存活快照/阶段；多rank分别计算后取最大卡。

```json
{
  "rank": 0,
  "phases": ["attention", "moe-dispatch", "moe-combine"],
  "coverage": {
    "unassessed_components": ["Attention workspace", "allocator extra"],
    "unassessed_phases": ["Dense", "logits", "MTP"]
  },
  "storages": [
    {"id": "shared-index-backing", "category": "persistent_buffer", "bytes": 67108864, "status": "assumed", "live_phases": ["*"]},
    {"id": "comm-backing", "category": "communication", "bytes": 1782579200, "status": "assumed", "live_phases": ["*"]},
    {"id": "graph-increment", "category": "graph_pool", "bytes": null, "status": "unknown", "live_phases": ["*"]},
    {"id": "temporary-KV-gather", "category": "activation", "bytes": 1048576, "status": "assumed", "live_phases": ["attention"]},
    {"id": "weight-backing", "category": "weights", "bytes": null, "status": "unknown", "live_phases": ["*"]},
    {"id": "kv-pool", "category": "kv_cache", "bytes": null, "status": "unknown", "live_phases": ["*"]}
  ]
}
```

这是演示schema，不是完整估算。数值均为示例条件，不能直接用于部署。

- id必须是唯一物理存储身份，同一backing的aliases合并到一条，live_phases取并集；脚本拒绝重复id，不自行推断alias。
- category支持activation、workspace、communication、graph_pool、persistent_buffer、runtime、allocator_extra、weights、kv_cache；后两项排除。
- bytes为非负整数字节或null。已知但依赖假设用assumed；无法给数值用unknown/null，不以0代替。confirmed也不意味着脚本验证过来源。
- live_phases列同时存活阶段；`["*"]`表示该账本全部阶段常驻，不能与其他阶段混列。更细生命周期可把一个阶段拆为多个快照；并发流水须显式给共存快照。
- coverage两个列表必须显式提供。无未覆盖项时才能写空列表；脚本不检查是否真的覆盖模型全部阶段。
- 若pool总额已包含tensor，不再同时添加tensor条目；reserved不得与allocated总值并列相加。分配器额外占用必须是去重后的slack。

输出保留每阶段已知小计及unknown列表。存在null或声明未覆盖项时，complete_ledger_peak_bytes保持null；脚本仍输出已知小计的max及对应阶段。unknown项可能改变真正峰值阶段。

`ledger_complete_as_declared=true`只表示输入清单自称完整，并非实测或正确性证明。conditional_numeric_storages列出依赖假设的数字；即使完整账有数值，也须以条件预算交付。
