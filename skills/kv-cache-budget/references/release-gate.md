# 单请求结构化校验

具体物理池需求报告使用此入口。概念解释或明确只求有效载荷时不扩大范围。

## 生成报告

1. 复制 [ledger-template.json](ledger-template.json) 为 `budget-reports/<case>/ledger.json`，填写单请求快照、完整组件、状态生命周期与物理布局。证据文件放在账本目录内，记录 SHA-256、行范围和固定后端 revision。
2. 运行 `python3 skills/kv-cache-budget/scripts/release_budget.py --input budget-reports/<case>/ledger.json --output-dir .budget-gate-output/<case>`。
3. 修复生成的缺项。独立复核者核对源码解释和组件覆盖，将 `review-request.json` 补为 `review.json`，填写实际 reviewer、decision 与 notes；不得由计算者改名自签。
4. 同一命令增加 `--review budget-reports/<case>/review.json`，退出 0 才交付生成的池需求。输入/证据变动后重新复核。

未通过时报告保留已知小计，`request_pool_need_bytes=null`；失败重跑替换旧产物。脚本不证明源码解释真实，也不能拦截任意聊天文字。CI 的 `KV budget gate / validate` 会重算已保存账本；强制合并检查需仓库管理员另配规则。

## 账本 v1 核心字段

| 字段 | 内容 |
|---|---|
| identity | model、model_revision、hardware、stage、scope_id、analyst、analysis_kind 与固定 backends revisions；不要求并行配置 |
| parameters | 模板中的单请求参数；confirmed/assumed、value、evidence_ids。假设仅用于 conditional 结果 |
| evidence | id、相对 path、sha256、line_start/line_end、kind、claim；source 另含 backend/revision |
| checks | backend_compatibility、active_path、inventory、lifecycle、allocator、admission、capacity、prefix_retention、speculation、async；含 verified、reason、source evidence |
| inventory | 实际独立 owner：id、owner、kind、owner_count、role(target/draft)，与 components 完整对应 |
| requests | 为兼容 v1 保留数组，但必须恰好一个请求：id、prefix_tokens、settled_tokens、reference_length、generated_tokens、evidence_ids、chunks=[{id,batch_id,tokens}] |
| components | owner 信息、attention_type、retention(full_history/window/fixed_state)、record_bytes、page_bytes；full_history 另含 compression_ratio |
| lifecycle | write、restore、retain、release、deduplicate；每项 reason、verified、source evidence |
| components[].requests | 仅对应上述单请求，parts 精确列 prefix、settled、每个 chunk:<id>、window_state、extra |
| parts | status、records、pages、formula、reason、evidence_ids；known 或有证据的 not_applicable(数量为0)；unknown 阻止总量发布 |
| layout | audit_layout.py 的真实 allocator 几何，组件页合计必须与本账一致 |
| cost_checks | allocator/admission/capacity 的 scope、bytes、source evidence，统一到同一 scope_id 的快照后比较 |

records/pages 为每 owner 数量，程序乘 owner_count。页按新增唯一归因计数，跨列边界只计一次；长历史记录用累计压缩差分。窗口与固定槽数量由生命周期取证，不由脚本猜测。

只检查本请求的实际 chunk 与配置上限，不汇总多请求，也不做跨卡归属。现有 v1 数组保留用于兼容，不表示支持多个请求。共享组件先按物理 backing 去重。

review 含 ledger_sha256、完整 evidence_hashes、reviewer、decision、notes。复核为声明而非身份认证。退出 0 表示结构/指纹/分账/声明成本及绑定复核通过，1 表示未完成，2 表示输入错误。observed_peak_bytes 始终 null；不生成实测峰值。

验证入口：`python3 -m unittest discover -s skills/kv-cache-budget/scripts -p 'test_*.py'`。必须覆盖组件遗漏、生命周期缺证据、Q/chunk 缺项、累计压缩边界、固定 state、重复请求拒绝、指纹与复核过期、成本冲突和失败重跑。
