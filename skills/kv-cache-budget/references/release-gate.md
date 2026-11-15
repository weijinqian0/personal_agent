# KV 交付门禁

适用于具体 KV 部署需求与逐 rank 容量报告。只询问概念或明确只求静态有效量时，保留原有范围，不强制扩大为部署预算。

## 能强制什么

`scripts/release_budget.py` 是报告的受控生成入口，标准库即可运行。缺证据、缺组件分账、成本冲突或无有效独立复核时，退出非零，`request_pool_need_bytes=null`，报告只显示已知组件小计和缺项。失败重跑会覆盖旧的成功报告。脚本不读取自由格式回答，也不能拦截 Agent 绕过入口手写的聊天内容。

GitHub Actions 的 `KV budget gate / validate` 会执行回归测试，并重新生成 `budget-reports/**/ledger.json` 对应的报告。只有全部通过才上传报告 artifact；没有已保存账本时明确报告未认证任何部署结果。手写 Markdown、旧的 `audit_layout.py` 成功退出、仓库中已有示例，都不是本门禁的通过凭据。

这些检查证明结构、文件指纹、声明的版本归属和算术一致性，不证明源码解释正确、文件来源真实或 reviewer 身份经过认证。不能把字符串 `status=verified`、`decision=approved` 当成独立事实。复核应由另一人或隔离的评估过程完成，检查证据含义和是否遗漏 owner；不得让计算者换个名字自行签署。回归测试使用的 synthetic reviewer 仅为测试夹具。

如果需要在 GitHub 合并层强制执行，由仓库管理员在 ruleset/branch protection 中将 `KV budget gate / validate` 设置为 required check，并配置独立的 PR review 权限。仅新增 workflow 不会自动开启分支保护，也不能保证绕过 GitHub 或受控入口的自由文本。现有仓库权限与保护规则不由本工具擅自修改。

## 工作流

1. 从 [ledger-template.json](ledger-template.json) 复制账本，在 `budget-reports/<case>/` 内保存 `ledger.json` 和必要的源文件/配置/日志快照。模板有意不完整，不能发布部署总数。
2. 填写版本、配置、组件 inventory、五项状态生命周期、每请求 P/Q/各批 chunk、物理页归因和实际 allocator。证据使用相对账本目录的路径、整个文件 SHA-256、行范围和 claim；源文件绑定后端 revision。
3. 运行：

   ```bash
   python3 skills/kv-cache-budget/scripts/release_budget.py \
     --input budget-reports/<case>/ledger.json \
     --output-dir .budget-gate-output/<case>
   ```

   未提供 review 时退出 1，生成 `report.md`、`result.json` 和 `review-request.json`。先修复证据/分账问题，不通过改状态或删组件消除错误。
4. 独立复核者检查真实源码和生命周期，而非只读计算者的结论。将 review-request 复制为账本旁的 `review.json`，记录实际 reviewer、`decision=approved` 与非空 notes，保留精确的 ledger hash 和 evidence_hashes。拒绝或未完成时保持 pending/rejected。这个动作不由 release 脚本自动执行。
5. 运行带 `--review budget-reports/<case>/review.json` 的同一命令。退出 0 才能交付生成的报告；输出末尾区分条件池需求与实际部署需求。任何账本/参数/证据修改都需要重新校验与复核，不沿用旧签署。
6. 提交账本、必要快照和 review，CI 用以下入口逐案检查：

   ```bash
   python3 skills/kv-cache-budget/scripts/check_saved_budgets.py \
     --root budget-reports --output-dir .budget-gate-output
   ```

不要提交包含访问凭据的启动环境或日志；证据快照只保留核算与复核所需内容。生成的 `.budget-gate-output/` 不作为输入提交，CI 总是重新生成。

## 账本格式 v1

| 字段 | 必填内容与检查 |
|---|---|
| schema_version | 整数 1 |
| identity | model、model_revision、hardware、role、rank、stage、analyst、scope_id、analysis_kind；backends 为后端名到固定 revision 的映射；不能使用 main/HEAD/latest 等浮动标识 |
| parameters | 模板列出的全部字段；每个含 status、value、evidence_ids；status 为 confirmed 或 assumed。assumed 只允许 analysis_kind=conditional，且生成报告明确列出假设，不冒充实际配置 |
| evidence | 每条 id、相对 path、sha256、line_start/line_end、kind、claim；source 还含 backend、revision，必须匹配 identity.backends。校验内容指纹与范围，拒绝路径或符号链接逃逸；每个后端均需 source 证据 |
| checks | backend_compatibility、active_path、inventory、lifecycle、allocator、admission、capacity、rank_ownership、prefix_retention、speculation、async；每项 status=verified、reason 和至少一份 source evidence。这里的 verified 仍是须复核的声明 |
| inventory | 独立从实际构造枚举的 id、owner、kind、owner_count、role(target/draft)，覆盖全部 target/draft KV、Index、SWA、压缩器等；物理 alias 合并后 owner 唯一。与 components 必须逐项匹配 |
| requests | 每请求 id、prefix_tokens P、settled_tokens Q、reference_length、generated_tokens、evidence_ids，以及有序 chunks=[{id,batch_id,tokens}]。程序重建每批位置，检查可用历史、prefix 对齐、每个公共 batch token 额度及在途批数 |
| components | inventory 同名信息＋attention_type、retention、record_bytes、page_bytes；full_history 还需 compression_ratio。retention 可为 full_history/window/fixed_state，不能默认所有state按chunk线性增长 |
| components[].lifecycle | write、restore、retain、release、deduplicate 五项；每项 status=verified、reason、source evidence_ids。恢复不适用或零增量也需来源和理由 |
| components[].requests | 每个请求一条 id 与 parts；不可漏请求或把两个未结算批合并 |
| parts | 精确包含 prefix、settled、每个 chunk:<id>、window_state、extra。每项 status、records、pages、formula、reason、evidence_ids。已知数用 known；有证据的不适用用 not_applicable 且 records/pages=0；未知用 unknown，不能发布总量 |
| layout | 原 audit_layout.py 输入，包括组件/页/组/真实 allocator 几何。其 components 必须与完整 P/Q/chunk/state 账的页面合计一致；不接受与账本无关的旧算例 |
| cost_checks | allocator/admission/capacity 各一条 scope、bytes、source evidence_ids；统一到 identity.scope_id 的同一请求/阶段后比较。不能拿不同 max_model_len 或不同 rank 的数强行比较 |

`records` 和 `pages` 是单 owner 的数量；脚本乘 owner_count，bytes 由脚本计算。页面是**新增、唯一归因的页数**，不是每列独立 ceil：跨列边界页只计一次。full_history 的 P、Q、每批 C_i 记录数按累计 `floor(x/compression_ratio)` 差分核验。窗口与 fixed_state 的页集合/版本数量由真实生命周期取证，并由独立复核检查；脚本不根据 W 自动猜测回收行为。

跨请求或组件的共享页必须先按物理身份去重，归入唯一计费 owner/条目，其余行引用生命周期证据。v1 的 full_history 记录守恒按请求核验；若复杂共享/重布局无法表示为唯一归因页账，保持 unknown，扩展适配器后再发布，不以重复求和或 padding 系数替代。

review.json 的字段为 `ledger_sha256`、完整 `evidence_hashes`、`reviewer`、`decision`、`notes`。review 与输入/证据指纹绑定；身份为复核声明，不是加密签名或 GitHub 身份认证。CI artifact 是受控工具的输出，不认证任意聊天文本。

## 退出与产物

- 0：结构、指纹、分账、声明成本与绑定复核均通过，生成条件/部署请求池需求。
- 1：缺项、未知、成本冲突或未完成有效复核；仅交付缺口和明确标注的已知组件小计。
- 2：JSON、类型或文件读取等输入错误；生成 blocked 产物，不保留先前成功总量。

`result.json` 包含 `report_sha256`；报告和 JSON 来自同一次重算。`allocated_pool_bytes` 只沿用 audit_layout 已核对范围的日志值；`observed_peak_bytes` 始终 null，本门禁不生成实测峰值。输出路径不能覆盖账本、review 或证据。使用 CI 的 exit code 和成功 artifact，不能只检查目录里是否存在 report.md。

## 必须保持的回归

漏state、漏Q/第二chunk/extra、未知冒充0、prefix恢复或释放证据缺失、两请求重复公共batch额度、混用后端revision、指纹变化、review过期、自签署、capacity/allocator冲突、页面账与物理映射不一致、失败重跑遗留成功报告，均须被拒绝。测试还保留 fixed_state 不随chunk盲乘和压缩记录累计边界守恒的正例。
