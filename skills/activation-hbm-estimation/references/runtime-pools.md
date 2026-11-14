# HCCL、图池与测量

## 通信常驻

逐rank列物理域：成员、有效配置、初始化状态、收发份数、共享/独占、分配bytes和来源。

```text
M_comm = sum(已初始化且去重的通信域实际buffer)
         + 独立MC2/symmetric/SHMEM等池
         + 未包含的链接/context/辅助资源
```

环境变量是默认值，options可覆盖。payload、通信次数、层数、peer数都不是池份数；进程组复用、独立MC2域、惰性初始化改变总额，取得communicator name也可能触发分配。

某些CANN/TorchNPU文档指定每域收发两份，其他版本表述不同，先核对：[CANN 8.2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/maintenref/envvar/envref_07_0080.html)、[CANN 9.2 beta2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_BUFFSIZE.md)。文档MB/M与实现1024²的换算也需核实，不静默混用十进制MB/MiB。

历史源码的get_hccl_config_for_pg_options对普通域、DP、EPLB和MC2使用不同规则；数值见版本化算例。其他版本重新解析，不复用200/50/400常数。

只切prefill时，decode/初始化仍可能保留MC2资源。不能因A2AV便扣除MC2；须证明整个部署没有分配才扣。

## 图池

区分capture前已有静态buffer、捕获后新增持久分配、捕获过程临时峰值、replay图池外新增存储。图池整体与池内tensor只计一遍；分块图跨层和多个descriptor的复用需证据。

FULL_DECODE_ONLY的prefill eager仍可能与decode图池共存。最多6条序列、3个推测token时，普通验证可达6×(1+3)=24个位置；不证明capture档位或图池字节，还需dispatcher、padding和MTP路径。

常见入口为capture_model、npugraph_memory_bytes。日志搜索：

```text
Graph capturing finished ... took ... GiB
... GiB for NPU graph memory
```

capture前后free差分是净变化代理，可能夹带同期初始化和allocator行为。同步设备、固定其他活动，检查基线是否已有图池。缺日志可算已知图内buffer并保留G，不填经验常数。

## 设备级对账

用户授权且有设备时，记录加载后、通信初始化后、warmup后、全部capture后、满载prefill/decode/混合阶段的同步allocated/reserved/device free与峰值。

- 权重/KV按实际storage/池剔除，不能只按checkpoint或有效token成本扣除。
- reserved含allocated，不相加；必要保留空闲块/碎片单列。
- non_torch可能含HCCL/ACL/driver/workspace，已包含就不重复加通信池。
- 图增量若已混入activation profiling先分离；框架150MiB等安全余量不是全部runtime开销的实测值。
- steady-state与启动/capture峰值分列。缓存分配器使释放tensor不代表设备free马上增加。
- 多rank取最大卡；分支对比固定负载、dtype、decode捕获路径和公共buffer边界。

仅在已去重时可表达：

```text
M_nonweight_nonKV = P_persistent
                    + max(A_attention,A_dense,A_moe,A_logits,A_mtp,...)
                    + allocator_extra
```

未覆盖阶段/组件意味着总峰值未知，不能把已知少数阶段的max冒充完整峰值。
