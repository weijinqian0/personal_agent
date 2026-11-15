# 物理布局与交付状态

遇到混合Attention或异构共享池时，先按[Hybrid与padding核算](hybrid-padding.md)分开计算页内/页对齐/组与bucket空槽；GDN状态预留份数不能直接充当物理池成本。正文必须展示各层差额，不能只在结尾写“另加padding”。

用于单请求物理池需求或日志对账。仅解释Attention结构、只求有效载荷时不强制完整部署映射。不要将本例的25槽、592000B或模型名推广为默认布局。

## 最少证据

| 层次 | 记录 | 通过条件 |
|---|---|---|
| 运行身份 | 模型revision、后端commit/相关dirty diff、硬件、最终参数及生效patch | 不混用checkout或参考实现；条件场景与实际部署区分 |
| owner与spec | 主KV、Index、SWA、各state、draft；dtype/scale、window、压缩率、logical/storage block、原始/填充页 | 所有实际owner覆盖，别名去重 |
| allocator | 独立页/共享池/多池/ring；group、bucket、tuple、draft尾槽、alignment、stride、alias | 追到真实backing大小和独立分配ID；不是只找到Spec |
| 请求需求 | P、Q、各C_i、F、窗口回收和额外保留 | 组件分账完整，新增/基线页按并集 |
| 成本与闭环 | 各分配单位bytes、块数；allocator/admission/capacity；同范围日志 | 三条路径成本一致；没有日志标未实测，冲突标未完成 |

## 分配器分支

- 独立页：逐物理owner的页数×页字节，相同backing的views只记一次。
- 统一共享池：先从实际bucket、槽数、尾槽及alignment推导U，再以去重后的独立group block数×U计算。未用bucket或层槽仍可能占空间。同组主KV与Index不各乘一次U。
- 多池/不同大小bucket：按各池实际单位分开核算，不强行套一个U。
- 固定ring：真实槽数/版本数×槽成本，不随F盲目线性增长。窗口分页state则按真实回收规则计算。

分配与admission、capacity不保证使用同一实现。分别跟踪：分配的槽数/stride，admission每组消耗及计费分母，capacity使用的max_model_len/F与比例。特别核对MTP附加槽在三条路径是否一致。日志复算一致只是算术证据，不证明运行commit和本地代码一致。

`registered_bytes/num_blocks`仅用于同执行范围、同池、同分配范围；分子需是唯一KV backing字节，不能带重复注册views、独立通信buffer或漏掉未注册storage。范围不足返回“无法验证”。没有日志时可完成源码推导布局，保留未测量状态。

## 输出口径

| 字段 | 含义 |
|---|---|
| payload_bytes | 有效记录，不含物理填充 |
| component_page_bytes | owner自身页成本，含页内padding，不一定含共享池空槽 |
| request_pool_need_bytes | 完整物理映射后的请求池需求；声明admission上界/实际持块 |
| allocated_pool_bytes | 初始化或动态分配的池供给，不能再叠加池内请求需求 |
| observed_peak_bytes | 有对应运行测量的峰值，静态推导不得填写 |

组件表完成、物理映射未完成时，主结论写“部署池需求尚未算完”，未知字段null/符号；禁止使用“约X加少量padding”把组件小计包装成部署总数。若仍缺prefix/投机/延迟保留，映射完成也不代表请求范围已闭合。

布局身份包含会改变布局的模型、后端代码、硬件、dtype/scale、block、Hybrid、draft及缓存重布局。相关身份变化重新核对；仅长度/chunk变化且布局不变时复用几何并重算需求。使用实际 owner shape，不推导多卡归属。

## 分组重放

从固定版本的实际分组函数生成最终 group/bucket 清单，保存调用输入、输出和源码位置；包含 approximate GCD 等自动拆组，不能用最大原始层数代替最终槽数。共享 tuple 布局向校验器提供 `grouping_evidence`、每组按 `page_buckets_bytes` 顺序排列的 `owners_per_bucket`（空 bucket 填0）及 `trailing_slots`。校验器按 `max(各组各bucket的owner数)+trailing_slots` 复算槽数；不适用此公式的分配器须另建实际映射。

## 算术核对器

`python3 scripts/audit_layout.py --input layout.json` 仅验证声明布局，不能自动读取模型、检查源码真实性、识别alias或测量峰值。输入示例见 `scripts/fixtures/shared-tuple-layout.json`。

支持两种可明确表达的模式：`shared_tuple`（U=slots×sum(bucket)，无额外alignment）和`independent_pages`（已去重独立页）。其他布局返回未支持/未完成，继续按实际allocator建账，不能套一个近似padding系数。日志、admission检查是可选的；提供后若有冲突会阻止该输入被标为布局完成。

`source_trace_complete`、`request_mapping_complete`和`snapshot_scope_confirmed`为调用者的证据声明，不是脚本独立验证的结论。必须附来源；缺失时不产出部署需求数值。额外checkpoint/投机/额外保留需求仍由专项流程检查。

## 回归算例与行为验证

本次已给定几何：block32、主记录608B/页19456B，Index132B/页4224B，25槽。H=128K、F=16K，组块数 `[1024,32,1034,8197,4129]`，组件页小计6667249792B，池需求8534272000B，差1867022208B。组件页小计不等于部署需求；此算例的H/F为独立条件预算，不声称带90%命中的单请求一定有16K剩余。

算术测试覆盖尾槽变化、admission冲突、日志范围未确认、独立GQA页、缺物理映射、多池未支持和单请求在途边界。共享/alias的自动发现不在脚本能力内，不能给未去重views直接贴上完成标签。

行为验证使用原始配置和用户请求，分别检查资料齐全与缺allocator资料时的结果：后者应交付组件小计并明确未完成，而不是输出部署总数。单元测试不能证明skill行为；独立评估者不应提前得到预期答案或本次缺陷结论。
