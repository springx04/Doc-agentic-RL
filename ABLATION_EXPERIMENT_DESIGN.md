# OpenClaw-RL / BayesTool-RL 消融实验设计（v1）

> 文档状态：实验设计冻结候选版，不执行实验，不修改项目代码。
> 审计日期：2026-08-10。
> 审计基线：当前工作树，HEAD <code>3ccfd2512193f788204ba589007698eceb59857c</code>。正式运行必须另外记录实际 Git SHA、工作树补丁哈希和配置哈希。
> 适用主模型：Qwen3-VL-4B。8B 仅作为规模复核，不和 4B 混入同一消融统计。
> 规范词：本文中的“必须”是结果可用于方法结论的硬条件；“建议”是默认实现；“可选”不属于核心消融。

## 1. 结论先行

推荐采用“1 个 Full + 7 个单组件消融 + 1 个 meta-credit package 消融 + 4 个锚点对照”的分层方案。核心结论来自 Full 与 8 个预注册 leave-one-out 对照；锚点对照用于排除“只是用了更多采样、更多扰动数据或不同奖励”的替代解释。A07 同时移除 meta batching 与 suffix credit，必须按 package effect 解释；若要拆分两者，需另做 2×2。

当前仓库**不能直接启动正式消融**。这不是方案不可执行，而是代码快照存在若干已定位、可验收的集成缺口：旧 launcher 的分组规模与新 loss 契约不一致，Stage-D 数据格式没有闭环，现有 shell 未透传全部消融参数，真实 4B runner 还依赖当前 checkout 中不存在的模块。本文把这些问题列为 Phase 0 硬门禁；后续执行 agent 只有在门禁全部通过后，才能进入正式训练。这样可以保证“跑出来的每一组”确实对应所声明的消融，而不是静默 no-op、空分组或算力不匹配。

主实验固定：

- 每题 4 个潜在工具世界（R=4）；
- 每个世界只选 1 个决策前缀，并生成 4 个同前缀 continuation（K=4）；
- 每题恰好 4 个合法 decision groups、16 条 policy records；
- 每个 optimizer step 使用 2 个完整问题，即 32 条 policy records；
- 工具预算 8、最大轮数 10、最大观察字符数 8192；
- 所有方法共享问题顺序、world spec、随机种子、生成预算、优化器和评测解码；
- 训练种子固定为 42、43、44；论文级确认再为全部核心组统一补 45、46；
- 内部 200 题测试集和四个公共 benchmark 只在方案、checkpoint 选择规则和分析脚本冻结后使用。

核心主指标是固定扰动世界上的 Robust Macro-ANLS；Healthy ANLS 是安全性指标，预注册非劣效界限为 −0.01。训练内的 Bayes utility 只作为优化诊断，不能替代官方评测分数。

## 2. 项目现状与实验边界

### 2.1 数据与评测入口

根据 <code>data/statistics.json</code>：

| 数据 | QA 数 | PDF 数 | 用途 |
|---|---:|---:|---|
| <code>data/train.jsonl</code> | 1,000 | 81 | 训练、内部 dev 和 risk calibration 的文档级划分来源 |
| <code>data/test.jsonl</code> | 200 | 17 | 最终 in-domain 测试；不得参与调参、选 checkpoint 或筛选消融 |

项目已有四个外部评测适配器，位于 <code>toolcall-rl/eval_benchmarks/</code>：

| Benchmark | 当前数据规模 | 官方主指标 |
|---|---:|---|
| DocVQA 2026 validation | 以准备后的 integrity manifest 为准 | Accuracy |
| LongDocURL public | 2,325 | Generalized Accuracy |
| MP-DocVQA validation | 5,187 | ANLS |
| DUDE validation | 6,315 | Overall ANLS |

外部数据量必须以正式运行时的 <code>evaluation_integrity.json</code> 为准，不允许用 README 中的数字替代运行期完整性检查。

### 2.2 当前方法组件与实现位置

| 组件 | 主要实现 |
|---|---|
| 工具世界、局部/上下文/共享/非平稳扰动 | <code>toolcall-rl/bayestool/world.py</code> |
| Belief filter 与 HBD | <code>toolcall-rl/bayestool/belief.py</code>、<code>toolcall-rl/train_bayestool_belief.py</code> |
| Q、risk、DVOI、regret、reopen | <code>toolcall-rl/bayestool/decision.py</code> |
| Utility、decision group、switch、pre-invariance | <code>toolcall-rl/bayestool/training.py</code> |
| Rollout、候选动作、证据状态、分支记录 | <code>toolcall-rl/generate_with_retool.py</code> |
| 问题级分组和动态 batch | <code>slime/slime/ray/rollout.py</code> |
| Auxiliary bundle 附着 | <code>slime/slime/rollout/sglang_rollout.py</code> |
| Megatron advantage/action signal | <code>slime/slime/backends/megatron_utils/loss.py</code> |
| Stage A | <code>toolcall-rl/run_bayestool_stage_a.py</code> |
| 训练审计 | <code>toolcall-rl/analyze_bayestool_training_log.py</code> |
| 外部评测与官方 scorer | <code>toolcall-rl/eval_benchmarks/</code> |

### 2.3 当前可用性分类

| 类别 | 项目 | 处理 |
|---|---|---|
| 已有底层开关 | DVOI、regret branching、reopen、meta episode、persistent belief；HBD trainer 有独立开关 | 仍需由统一 launcher 显式透传并做 manipulation check |
| Megatron loss 已有关闭守卫 | switch loss、pre-invariance | flag 到达 Megatron model 后会跳过对应 auxiliary loss；rollout 仍会附着 bundle，需补 launcher 透传、消费计数与计算开销报告 |
| 仅需补实验入口 | 所有 opt-out 的 <code>ABLATION_ID → CLI</code> 映射、resolved config dump | 新 launcher 集中实现，禁止每次手改 shell |
| 必须补集成 | R4×K4 分组、Stage-D meta JSONL、checkpoint manifest 校验、Megatron question weight | Phase 0 完成并测试 |
| 只能做诊断、不能做主 Full | heuristic belief/Q/risk | 只用于 smoke 或能力替代消融 |

## 3. 研究问题和可证伪假设

### RQ1：完整 BayesTool 的提升是否超出扰动训练、奖励塑形和额外采样本身？

- H1a：Full 的 Robust Macro-ANLS 高于计算匹配的 neutral-belief baseline。
- H1b：Full 相对 healthy-only GRPO 和 domain-randomized outcome-GRPO 的收益主要集中在 perturbed 和 change worlds。
- H1c：Full 的 Healthy ANLS 相对最强计算匹配 baseline 不劣于 0.01。

### RQ2：信念学习与两类正则各自解决什么问题？

- H2a：去掉 HBD 将恶化 held-out filter NLL、Brier、ECE 和分层状态识别率。
- H2b：去掉 switch loss 将降低区分观测之后的 belief-switch accuracy。
- H2c：去掉 pre-invariance 将增大区分信息出现前的 action-distribution JS；它不应被解释为一般时序不变性，因为当前实现更接近 root-prefix regularizer。

### RQ3：DVOI 与 regret-guided branching 是否改善决策而非仅增加计算？

- H3a：去掉 DVOI 将提高 relative oracle regret，并降低有效诊断动作的命中率。
- H3b：regret-guided node selection 相对固定 root 或 seeded-random node，在保持相同 K、分支深度和 token 数时提高 Robust Macro-ANLS。

### RQ4：reopen 是否能恢复非平稳世界，而不是频繁重置？

- H4：去掉 reopen 主要损害 abrupt/gradual change worlds，并增加 change detection delay 和恢复调用数；healthy world 的 reopen false-positive rate 应接近零。

### RQ5：meta episode 与跨题 belief 持久化是否分别有效？

- H5a：meta episode/suffix credit 改善同一 session 后续问题的策略学习。
- H5b：在保持相同 2–4 题序列、world 和 return 规则时，去掉 persistent belief 将降低 Q2–Q4 得分或增加工具调用。

## 4. 因果对照结构

建议按下面的 checkpoint 图执行，避免不必要地重复 Stage A，同时不把已经受某组件影响的 actor checkpoint错误地复用于该组件的消融。

~~~mermaid
flowchart LR
    W0["共同 tool-call warm-start W0"]
    AF["Stage A: Full belief / Q / risk"]
    AH["Stage A: no-HBD belief / Q / risk"]
    BF["Full Stage B"]
    CF["Full Stage C"]
    DF["F0 Full Stage D"]
    AM["A07 no-meta-credit package Stage D"]
    AP["A08 no-persistent-belief Stage D"]
    AX["A02-A06: 从 W0 分别运行自己的 B/C/D"]
    HBD["A01: 使用 no-HBD artifacts 运行 B/C/D"]
    BASE["B01-B03: 从 W0 独立运行匹配预算"]

    W0 --> BF --> CF --> DF
    CF --> AM
    CF --> AP
    W0 --> AX
    W0 --> HBD
    W0 --> BASE
    AF --> BF
    AF --> AX
    AH --> HBD
~~~

复用规则：

1. Full、A02–A08 可共享同一 seed 对应的 Full Stage-A artifacts。
2. A01 no-HBD 必须独立训练 belief filter，并基于该表示重新拟合 Q 和 risk；不能只在 policy 阶段隐藏 HBD 标签。
3. A02–A06 对 Stage B 或 Stage C 已可能产生作用，因此必须从共同 actor warm-start W0 开始各自运行 B/C/D，不能从 Full Stage-B checkpoint 才关闭。
4. A07 和 A08 的差异只在 Stage D 出现，可以从同一 Full Stage-C checkpoint 分叉。
5. 同一 seed 下的所有分叉必须使用相同数据顺序、world schedule 和 optimizer schedule。
6. B01–B03 从同一 W0 独立训练完整匹配预算；B00 只评测 W0。

## 5. Phase 0：正式实验前的硬门禁

### G0.0 冻结共同 warm-start W0

当前仓库没有已验证、可直接认定为 W0 的 tool-call checkpoint；原始 Qwen3-VL 也存在协议坍缩风险。正式消融前必须先指定或训练一个共同 W0：

- 只使用 <code>train_docs</code> 的 SFT/tool-protocol 数据训练，不得使用 dev/calibration/internal-test/公共 benchmark 的问题或答案；
- 固定训练配方、step、模型/processor 版本和 prompt/tool schema；
- 在 32 个 dev smoke 问题上要求 protocol-valid ≥95%、tool-call rate ≥80%、completion ≥75%，且无系统性 JSON/schema 崩溃；
- 保存 checkpoint SHA256、来源数据 hash、训练 Git SHA、配置和能力报告；
- B01、B02、B03、F0、A01–A06 都必须从同一个 seed-paired W0 独立分叉；A07/A08 按图从 Full Stage-C 分叉；
- 若 W0 需要多个训练 seed，则各方法在同一 seed 下共享对应 W0；不得让 Full 使用更强的 warm-start。

这些阈值只是 actor 能进入 RL 的能力门禁，不是最终性能结果。W0 未通过时先修复 warm-start，不得用 heuristic artifact 掩盖协议坍缩。

### G0.1 统一、自包含 launcher

后续执行 agent 应新增一个消融专用入口，例如：

<code>toolcall-rl/run_qwen3_vl_4b_bayestool_ablation.py</code>

该名称只是建议，当前文档不创建它。入口必须满足：

- 不依赖仓库中不存在的 <code>run_qwen3_vl_4b_smoke.py</code>；
- 接收 <code>--ablation-id</code>、stage、seed、训练/评测数据、基础模型、load checkpoint、输出目录；
- 使用固定映射表生成参数，不允许通过手改 shell 区分实验组；
- 启动前写出 <code>resolved_config.json</code>，包含所有 BayesTool、生成、batch、optimizer 和资源参数；
- 对未知 ID、互斥参数、缺失 artifact、输出目录已存在一律 fail closed；
- 正式默认值使用 Stage C 的 branch probability 0.25、regret threshold 0.08，不能沿用当前 real-test runner 的强制分支 1.0/−1.0；
- 明确区分 HF base model 与 actor load checkpoint。评测 wrapper 不能把两者都塞入一个含义模糊的 <code>--model</code> 参数。

当前两个入口均不可直接视为正式 launcher：

- <code>retool_qwen3_vl_4b_bayestool_rl.sh</code> 固定 4 worlds × 2 replicas；
- <code>run_qwen3_vl_4b_real_docvqa_test_05.py</code> 当前 checkout 缺少其 import 的模块，且默认 artifact 路径不存在。

### G0.2 统一 R4×K4 分组契约

正式主配置必须实现：

1. 每题生成 R=4 个 answer-independent latent worlds。
2. 每个 world 只保留一个被选 decision prefix。
3. 从该 prefix 生成 K=4 个同前缀 continuation。
4. 每个 decision group 只含同一 <code>question_id</code>、<code>latent_world_id</code> 和非空 <code>decision_prefix_id</code>。
5. 每题恰好 4 个合法 group，共 16 条 policy records。
6. 父轨迹和 children 不得重复进入同一 group，不得形成 K 大于 8 的超大组。
7. 关闭 regret-guided branching 时，改用 fixed-root 或 seeded-random prefix，但仍生成 K4；不能退回 K1/K2。
8. 将“是否采用 regret-selected prefix”和“是否构造训练 group”解耦：Full 中概率门通过且 eligible 时使用 regret prefix；概率门未通过、无 eligible node 或 horizon 不满足时，必须从固定 root 生成 K4 fallback。Stage-B/C/D 的 0.10/0.25/0.25 只控制 regret-prefix 曝光率，不能控制 group 是否存在。

这与当前 <code>_bayes_group_is_valid()</code> 的 K∈{4,8} 以及 <code>_prepare_bayes_question_batches()</code> 的每题 4–6 groups 契约一致。主实验固定 4 groups；5–6 groups 只留给后续敏感性分析。

### G0.3 所有 opt-out 必须做端到端 manipulation check

仅检查 config 值不够。每个运行都必须写 <code>feature_activation.json</code>，至少包含：

| 机制 | Full 应满足 | 对应消融应满足 |
|---|---:|---:|
| HBD | HBD loss/metric 有限，checkpoint 标记 enabled | HBD loss contribution=0，manifest 标记 disabled |
| switch | attached/consumed bundle count &gt; 0，loss contribution &gt; 0 | consumed count=0，loss contribution=0；attached count 单独报告 |
| pre-invariance | attached/consumed bundle count &gt; 0，loss contribution &gt; 0 | consumed count=0，loss contribution=0；attached count 单独报告 |
| DVOI | eligible count &gt; 0 且至少一次实际选择 | actual selection count=0 |
| regret branching | regret-selected group count &gt; 0 | regret-selected count=0，但 K4 group coverage 不下降 |
| reopen | change worlds 中 reopen count &gt; 0 | reopen count=0，surprise/change detector 仍记录 |
| meta-credit package | 2–4 题 episode count &gt; 0 且 suffix-credit count &gt; 0 | meta episode=0，suffix-credit=0，flat session 行数匹配 |
| persistent belief | belief carry count &gt; 0 | belief carry=0、逐题 belief reset 匹配；非 belief cache carry 与 Full 一致 |

当前 Megatron 路径的 <code>_normalise_bayestool_aux_bundles()</code> 会根据两个 opt-out flag 跳过对应 loss，因此算法开关是有效的；但 rollout 侧仍会构造/附着 bundle。正式运行必须确认 flag 被 launcher 透传、Megatron consumed count 为 0，并把对应权重置 0 作为冗余保护。若主张严格训练计算匹配，应在构造端也加 guard 或使用等量 dummy bundle，并分别报告 actor FLOPs 与 rollout CPU/内存开销。

### G0.4 Stage-D 数据契约闭环

当前 builder 的 <code>--meta-output</code> 生成顶层 <code>questions</code> 数组，而 runtime 读取每行 <code>metadata.meta_questions</code>。执行前必须生成真正可被 runtime 消费的 RL JSONL，并验证：

- 每行都有稳定的 <code>meta_episode_id</code>、<code>session_id</code> 和 2–4 个 <code>meta_questions</code>；
- 一个 meta episode 内的问题来自同一文档/session；
- 每个子问题保留唯一 <code>question_id</code>、答案、metric、document hash 和 event slice；
- session world spec、切换时刻和 observation seeds 固定；
- Full、no-meta、no-persistence 使用完全相同的问题序列；
- no-meta 将同一序列展平为单题更新并保留 session 边界；持久 belief 仍需跨展平行传递，该组按“meta batching + suffix credit package”解释；
- no-persistence 保留 meta episode、suffix return、序列顺序以及文档/工具/结果 cache，只在问题边界清空 belief/filter hidden state。

如果当前实现无法在 no-meta 时保留跨行 session belief，则 A07 会进一步混入 session persistence，运行无效；必须先补解耦接口。若要把 A07 package 再拆成两个因果因素，执行 batching×suffix-credit 2×2。

### G0.5 Stage-A artifact 真实性与校准

Stage C/D 的 capability gate 当前主要检查路径存在，正式入口必须额外验证：

- belief/Q/risk 文件可反序列化并能完成一次前向；
- feature schema、feature name/order、模型版本和 shape 一致；
- artifact 的 Git SHA、训练数据 hash、document split hash 和 config hash 匹配；
- belief filter 在 held-out 文档上的 NLL/Brier/ECE；
- Q head 的 held-out ranking/回归指标；
- risk calibrator 只使用文档不重叠的 calibration split，报告 Brier/ECE；
- artifact SHA256 写入 actor run manifest。

不得使用 <code>run_bayestool_stage_a.py --fit-risk</code> 的同集拟合方式作为正式 risk calibration。建议先生成 document-disjoint risk-validation JSONL，再直接调用 trainer，或由执行 agent补齐 wrapper 参数。

### G0.6 去除模拟器近 oracle 捷径

当前公开 observation 中的 <code>semantic_agreement</code> 会由 <code>corruption_applied</code> 映射为 1.0/0.80/0.65，<code>status/error_family</code> 也包含由注入器产生的强提示。这可能让 belief filter 学会读取模拟器标签，而不是从可观察结果推断可靠性。

主实验必须二选一：

1. 推荐：policy/belief 的公开特征中移除 <code>corruption_applied</code>、注入式 <code>error_family</code> 和由它们直接计算的 agreement，agreement 改为由候选输出、证据一致性和实际错误返回计算；latent 标签只用于监督、审计和离线指标。
2. 若暂时保留：所有结果明确标记为 simulator-assisted，并把“observable-only features”升级为必做消融，不能外推到真实工具故障。

执行后必须自动断言 model-visible JSON 不含 latent world type、corruption flag、gold page、gold answer 或注入器内部状态。

### G0.7 Megatron/FSDP 后端一致性

主实验选定一个后端并全程固定。当前 Qwen3-VL launcher 偏向 Megatron，因此至少增加以下集成验证：

- <code>bayes_loss_weights</code> 在实际 loss 中被消费，等问题权重数值测试通过；
- <code>bayes_grouping_report</code> 不会作为未知 scalar dict 使 rollout logger 抛错；
- action reward override 的 token count、abs sum 和梯度均非零；
- question-count checkpoint 若后端不支持，则统一使用固定 optimizer-step 的 <code>--save-interval</code>；
- 同一 frozen batch 在 FSDP 与 Megatron 的 group baseline/weight 计算一致到预设容差。

### G0.8 32 题真实 smoke

使用 32 题、至少 16 个 PDF、R4×K4，共 512 条 policy continuations。通过条件：

| 检查 | 必须满足 |
|---|---|
| 分组 | 每题 4 groups；group size 仅为 4；跨 question/world/prefix 计数全为 0 |
| world | 4 个 slot 覆盖 healthy、local/context、shared、change；healthy corruption 数为 0 |
| 工具 | 至少一次真实工具调用、一次视觉输入；结构化 corruption 解析失败为 0 |
| 信号 | reward/utility 有方差；非零 advantage；有限非零梯度；至少一次 optimizer step |
| 辅助机制 | Full 的 switch/preinv bundle 非零；DVOI/branch 有曝光；change 子集中 reopen 有曝光 |
| 产物 | checkpoint、dump details、resolved config、run manifest、activation report 均存在 |
| 审计 | <code>analyze_bayestool_training_log.py --strict</code> 返回 <code>effective</code> |

严格审计只是必要条件；若任一机制没有曝光，即使 strict 返回 effective，也标记为 <code>not_exercised</code>，不得进入正式矩阵。

## 6. 核心消融矩阵

### 6.1 Full、七个单组件消融与一个 package 消融

| ID | 组名 | 相对 Full 的唯一因子 | 精确实现要求 | 计算/数据匹配 | 主要机制指标 |
|---|---|---|---|---|---|
| F0 | Full | 无 | learned HBD + switch + pre-invariance + DVOI + regret node selection + reopen + meta + persistent belief | R4K4，标准预算 | 全部 activation 非零 |
| A01 | −HBD | belief filter 不使用 HBD objective | 直接 trainer 使用 <code>--without-hbd</code>；相同架构、replay、split、epoch、seed；随后重拟合 Q/risk | belief 参数量与训练步数相同 | filter NLL/Brier/ECE、state accuracy |
| A02 | −Switch | 无 switch auxiliary objective | Megatron consumption 关闭；CLI flag + weight=0；pre-invariance 保留；attached bundle 可保留但必须计数 | consumed aux forward=0；另报 rollout-side 开销 | belief-switch accuracy |
| A03 | −PreInv | 无 pre-invariance objective | Megatron consumption 关闭；CLI flag + weight=0；switch 保留；attached bundle 可保留但必须计数 | 同 A02 | pre-information action JS |
| A04 | −DVOI | 不用 DVOI 选择诊断动作 | <code>--bayestool-without-dvoi</code>；保留 candidates=4、Q/risk、belief prompt、R/K | policy records、candidate 生成上限相同 | relative oracle regret、probe precision |
| A05 | −RegretBranch | 不用 regret 选择 branch prefix | 将 prefix selector 替为 fixed-root；补充 seeded-random-prefix placebo；仍从所选 prefix 生成 K4 | group 数、K、horizon、generation tokens 相同 | branch regret、group return spread |
| A06 | −Reopen | change 后不重开 belief | <code>--bayestool-without-reopen</code>；surprise/change detector 继续计算和记录但不 reset | world、切换时刻、调用预算相同 | change delay、recovery calls、false reopen |
| A07 | −MetaCredit package | 不使用 meta batching 与 suffix credit | Stage-D 同序列展平为单题 return；保留 session ID 与 persistent belief | 相同问题、顺序、world、question units、policy records | package effect；Q1→Q2–Q4 增益 |
| A08 | −PersistBelief | 题间不继承 belief/filter hidden state | meta episode、2–4 题序列、suffix return及文档/工具/结果 cache 全保留，只 reset belief state | 相同序列、cache 与计算量 | carry/reset count、later-question ANLS/calls |

注意：

- A02/A03 的 Megatron loss 守卫已存在，但现有 shell 没有统一透传；只有在 G0.3 的 consumed-count/loss check 通过后才可纳入正式结果。
- A05 不能直接把 branching 全关掉后接受空 group；那会同时消融训练数据量和 advantage 结构。
- A07/A08 必须作为 Stage-D 分叉。A07 是预先声明的 meta-credit package effect；如需分别归因于 batching 与 suffix credit，补做 <code>batching on/off × suffix return on/off</code> 2×2。A08 只能 reset belief/filter hidden state，不得同时清空文档、工具结果或视觉 cache。
- A01 改变 belief representation 后，Q/risk 必须重拟合，否则会把接口错配误当作 HBD 效果。

### 6.2 必做锚点对照

| ID | 对照 | 目的 | 实现要点 |
|---|---|---|---|
| B00 | Warm-start, no RL | 衡量全部 RL 增益 | 仅评测共同 W0 |
| B01 | Healthy outcome-GRPO | 普通 agentic RL 基线 | 每题 4 个独立 fixed-healthy slots、每 slot K4；官方任务 reward；保持 4 groups/16 records 和 token 预算 |
| B02 | Domain-randomized outcome-GRPO | 排除“只需见过扰动” | 使用相同 world schedule，但 policy 不见 belief block，只优化 outcome reward |
| B03 | Compute-matched neutral-belief utility | 排除额外候选、结构化提示和 utility 的作用 | 相同 R4K4、trajectory utility、action signal、等长固定 prior block；不更新 belief、不用 DVOI/regret/reopen/meta |

省略 <code>--bayestool-enable</code>（即内部 <code>bayestool_enable=False</code>）不能自动等同 B03，因为当前非 Bayes 路径可能同时改变 reward、process signal、grouping 和提示。B03 需要一个明确的 matched profile。

### 6.3 机制 placebo 与敏感性分析

这些不进入八个核心比较的 Holm 家族，但用于解释机制：

| ID | 分析 | 设计 |
|---|---|---|
| P01 | Belief placebo | 等长 belief block，按 question 内 seeded shuffle belief；检测“提示存在”而非信念正确性 |
| P02 | DVOI placebo | 与 Full 匹配诊断动作次数和成本，随机选 eligible probe |
| P03 | Branch placebo | 相同 K/horizon/token，使用 seeded-random eligible prefix |
| P04 | Reopen placebo | 匹配 Full 的 reopen 频率，在预先固定的非证据时刻 reset |
| S01 | Switch×PreInv 2×2 | 两者全开、仅 switch、仅 pre-inv、全关；仅在核心实验显示交互迹象后执行 |
| S02 | R/K | 主配置 R4K4；后续单独比较 K8 或 R=5/6，不与所有组件全交叉 |
| S03 | Belief particles | 4/8/16，其他不变 |
| S04 | Tool budget | 4/8/12，主结果固定 8 |
| S05 | Utility weights | outcome-only 与默认 0.15/0.20/0.35；先修复 resolved config 真正进入 reward |
| S06 | Observable-only | 移除模拟器衍生信号；若 G0.6 未用于主 Full，则本项升级为必做 |
| S07 | Meta decomposition | <code>batching on/off × suffix credit on/off</code> 2×2；persistent belief 固定开启 |

动作奖励和 evidence guard 存在强交互，建议另做一个定向 2×2，而不是和八个组件全因子：

- action reward scale：1 vs 0；
- evidence guard：full relation vs page-seen-only。

当前 action penalty 为硬编码，guard 也没有统一 mode 开关；后续 agent 需新增单一缩放系数和 guard mode。不要为了做这个二级分析改动核心实验定义。

## 7. 必须冻结的配置

### 7.1 算法与环境常量

| 参数 | 主值 |
|---|---:|
| Model | Qwen3-VL-4B tool-call warm-start |
| Worlds R | 4 |
| Siblings K | 4 |
| Questions/optimizer step | 2 |
| Policy records/step | 32 |
| Particles | 8 |
| Candidates | 4 |
| Max siblings | 4 |
| Branch horizon | 3 |
| Stage-B branch probability | 0.10 |
| Stage-C/D branch probability | 0.25 |
| Consensus threshold | 0.75 |
| Regret threshold | 0.08 |
| Local/context/shared surprise thresholds | 4/6/8 |
| Change threshold | 0.80 |
| Belief prompt token cap | 1200 |
| Tool-call budget | 8 |
| Max turns | 10 |
| Max observation chars | 8192 |
| Utility cost/inefficiency/failure weights | 0.15/0.20/0.35 |
| Switch/pre-invariance weights | 0.20/0.05 |
| Auxiliary interval | 2 |
| KL coefficient | 0.01 |

当前实现的 document quality reward 实际是约 <code>clamp(2.2 × q − 1)</code>，并可能叠加格式扣分；部分文档仍写 <code>2q−1</code>。正式实验不在消融中改这个实现，但必须在 resolved config/方法描述中如实记录。官方 ANLS/Accuracy 始终由独立 scorer 计算。

默认 Bayes utility 另按 <code>(2 × quality − 1) − 0.15C − 0.20I − 0.35F</code> 计算，其中 cost 的调用/延迟/文本/图像权重为 0.40/0.25/0.20/0.15，inefficiency 由 duplicate、no-gain 和 unnecessary calls 占比构成。动作级信号固定为 premature final −0.5、invalid protocol/tool args −0.5、超预算额外调用 −0.1。核心消融全部冻结这些值；任何 weight sweep 都只能进入 S05 或动作奖励二级分析。

switch/pre-invariance 的默认配对过滤也必须冻结：max OOD 0.15、belief JS 区间 0.10–0.80、最小 action margin 0.05、pre-invariance max JS 0.02。

### 7.2 运行级冻结项

每个 seed 的所有组必须一致：

- 基础模型、processor、视觉塔和 actor 初始权重；
- document split、question order、meta episode composition；
- <code>coupling_id</code>、<code>rollout_id</code>、<code>latent_world_id</code>、world type、故障强度、目标工具、切换时刻、observation seed；
- temperature、top-p、max new tokens、候选数、K、horizon；
- optimizer、学习率、warmup、clip、KL、global/micro batch 和梯度累积；
- prompt 模板、belief block 的最大长度、tool schema、缓存和并发；
- 训练的 attempted question schedule、optimizer updates 与 generated-token 预算；
- checkpoint 选择规则、失败重试规则、硬件类型和 GPU 数；
- 评测解码、world manifest、官方分母和“一题一个最终答案”协议。

如果机器资源不同，可以整体修改硬件或吞吐参数，但必须对全部组同步修改并生成新的实验版本号。建议主研究固定 4×A100：2 actor GPU + 2 rollout GPU、TP=2。

## 8. 数据、世界与泄漏控制

### 8.1 文档级划分

对 81 个训练 PDF 使用固定 <code>split_seed=42</code> 和 document hash，生成约 80/10/10 的：

- <code>train_docs</code>：Stage-A replay 与 Stage-B/C/D actor 训练；
- <code>dev_docs</code>：engineering gate、训练曲线和预先规定的 checkpoint 规则；
- <code>calibration_docs</code>：risk calibration 和 belief calibration。

不得按 QA 行随机划分。每个 split 输出：

- document hash 列表；
- QA 数和 PDF 数；
- source manifest hash；
- 与内部 test、四个 benchmark 的精确 hash、感知 hash、问题文本近重复审计。

<code>data/test.jsonl</code> 的 17 个 PDF 不参与任何训练期 eval。当前 launcher 若周期性评测该文件，正式配置必须改为 dev split；否则该 200 题只能降级为 validation，不能再报告为无偏 test。

### 8.2 Stage-A replay

正式目标：

- 至少 50,000 trajectories 或 200,000 tool events；
- 只来自 <code>train_docs</code>；
- replay 必须从完整 <code>dump_details/rollout_data/*.pt</code> 或等价的完整 interaction exporter 生成，不能用日志中稀疏的 first/last sample 代替；
- 以 document hash 做 train/validation/calibration 隔离；
- Full 与 no-HBD 共享完全相同的 replay 行、顺序、batch 和 seed；
- risk-validation 只来自 <code>calibration_docs</code>；
- 输出 belief、Q、risk checkpoint 和 capability manifest。

### 8.3 固定评测 worlds

为每个评测问题预生成 8 个 answer-independent <code>fixed_world_specs</code>：

1. healthy；
2. single_tool_degradation；
3. context_degradation；
4. shared_family_fault / text_core；
5. shared_family_fault / render_core；
6. shared_family_fault / structure_core；
7. abrupt_change；
8. gradual_change。

生成原则：

- 只使用 question/document metadata、固定 seed 和公开工具能力，不能使用 gold answer、gold page 或 rollout reward；
- 每个 spec 保存 <code>world_spec_hash</code>；
- 全部方法、seed 和 checkpoint 使用同一 manifest；
- healthy 必须断言 <code>corruption_applied_count=0</code>；
- abrupt/gradual 的 change point、目标 family 和强度固定；
- context fault 必须在预定工具路径上实际可触发，否则重新生成 manifest，而不是运行后按结果挑选。

当前 eval-only 默认一题一个 sample，sample/world slot 可能被 shuffle，因此不能把默认 slot 当作 healthy。应给每个 world 单独构造显式 <code>metadata.world_type</code> 或 <code>fixed_world_specs</code> 的 eval JSONL。

## 9. 训练预算与运行阶段

### 9.1 工程验证，不产生论文结论

| 阶段 | 规模 | 目的 |
|---|---|---|
| Unit/integration | synthetic frozen batch | 检查 schema、group、loss weight、opt-out |
| Full smoke | 32 题、≥16 PDF、512 continuations | G0.8 |
| Variant manipulation smoke | 每个消融 8 个定向问题 | 确认 Full 非零、消融为零且其余组件不变 |
| One-seed engineering run | 每组 256 question units、4,096 records | 估算显存、吞吐、激活覆盖；不得用于最终显著性 |

### 9.2 核心训练预算

每个独立 actor 训练分支冻结同一组 2,000 个 attempted question units；不得按各组成功情况补采不同问题：

| Stage | 比例 | Attempted question units | 目标 optimizer updates | Scheduled record slots |
|---|---:|---:|---:|---:|
| B | 20% | 400 | 200 | 6,400 |
| C | 60% | 1,200 | 600 | 19,200 |
| D | 20% | 400 | 200 | 6,400 |
| 合计 | 100% | 2,000 | 1,000 | 32,000 |

计数和 ITT 规则：

- 一个计划子问题算一个 attempted question unit；meta episode 的 2–4 个子问题分别计数。
- 每个 attempted unit 预留 4 个 K4 groups，即 16 record slots，并在所有组使用同一 question/world schedule。
- 模型协议错误、提前 final 或错误工具调用仍是有效 ITT 模型结果，保留负 signal，不得因“没有好轨迹”补采。
- group size、cross-ID、prefix 或组件 bundle 契约错误属于实现失败，允许值为 0；出现即停止该 run。
- 明确的 real-infrastructure failure 只对原 question/world 原地重试一次；仍失败时写入零权重 placeholder，保持 schedule 对齐，不用新问题替换。
- infrastructure placeholder rate 必须 ≤1%，且相对 Full 的差异 ≤0.5 个百分点；否则相关配对 run 全部作废。
- 动态 batch 应同时命中 attempted units 和 optimizer updates；若 placeholder 导致 update 无有效梯度，记录 skipped update，偏差必须小于 1%，并对全部配对组采用相同停止点。
- 同时报告 attempted、model-valid、infrastructure-placeholder、gradient-contributing 数量和 GPU-hours。
- 不允许按 test/benchmark 结果 early stop。主 checkpoint 固定为该 stage 最后一个完整 update；dev-best 只可作为预注册的次要报告。

### 9.3 种子与算力层级

推荐默认：

1. 核心矩阵 F0+A01–A08 先完成配对种子 42、43、44。
2. 若用于投稿级“显著/无损”结论，必须为**全部九组**统一补 45、46；Full−B03 healthy 非劣检验还必须给 B03 同步补 45、46，不能只给表现好的组补种子。
3. B00 无训练 seed；B01–B03 至少运行 42、43、44。
4. 8B 只复现 F0 与 B03，至少 3 个种子，作为规模外推，不并入 4B 的主检验。

按复用规则，三 seed 核心矩阵的 scheduled record slots 为 710,400：每 seed 的 F0、A01–A06 各 32,000，A07/A08 各只新增 6,400。B01–B03 三 seed 再增加 288,000，因此最低正式 actor 训练总预算为 998,400 scheduled record slots，不含 Stage A、smoke、评测和 infrastructure placeholders。所有运行先经过 256-question engineering phase，避免在无效配置上消耗全量算力。

## 10. 评测协议

### 10.1 内部固定世界评测

F0、A01–A08、B00–B03 的每个训练 seed 都先在静态固定世界套件上评测：

- 200 test questions；
- 8 fixed worlds；
- 共 1,600 question-world cases；
- 同一 eval decoding seed 和生成参数；
- 每 case 一个最终答案。

该套件衡量单题能力，但不能单独验证 A07/A08；两组的主机制结论来自下一节的顺序 session 评测。

主指标：

1. <code>Robust Macro-ANLS</code>：先对每个 fault world 求 ANLS，再对 7 个 fault worlds 等权平均。
2. <code>Healthy ANLS</code>：healthy world 的 ANLS，作为非劣效指标。

关键次指标：

- Non-stationary ANLS：abrupt 与 gradual 等权平均；
- Worst-world ANLS：7 个 fault-world ANLS 的最小值；
- completion、protocol valid、premature final；
- valid/total tool calls、duplicate/no-gain calls；
- supported-correct、unsupported-answer、evidence alignment；
- pages visited、render/OCR/crop 使用率和 image token cost；
- generated tokens、latency、GPU-hours；
- infrastructure-invalid rate 与 ITT failure rate。

### 10.2 Meta-aware 顺序评测

从内部 200-test 的 17 个 PDF 按稳定 question ID 构造每文档 2–4 题的固定 episode，并对同一组 episode 分别运行前述 8 个 fixed worlds；不足 2 题的文档不进入 meta 指标，但仍保留在单题 ITT 评测。manifest 必须固定：

- <code>meta_episode_id</code>、<code>session_id</code>、题目顺序和题位；
- 每个 episode 的 world spec、change point 和 observation seed；
- 独立状态键 <code>(checkpoint, training_seed, meta_episode_id, world_spec_hash)</code>；
- episode 内跨题维护文档/工具/结果 cache；
- F0/A07 跨题维护 belief，A08 仅在题边界 reset belief/filter hidden state；
- 每个子题仍产生一个独立 prediction，并以原始 question ID 交给官方/ANLS scorer。

主要报告：

- Q1 与 Q2–Q4 分层 ANLS、calls、tokens；
- <code>later_question_gain = mean(Q2–Q4) − Q1</code>；
- belief carry/reset count；
- change 前后题位的恢复延迟；
- 相同 episode/world 下 F0−A07、F0−A08 的 paired document effect。

若公共 benchmark adapter 只生成独立问题、没有 session/meta schema，则它不能检验 persistence。只有在能按同一文档构造固定 2–4 题 episode、维护上述状态生命周期并逐子题导出官方 prediction 后，才能在公共 benchmark 报告 A07/A08 的跨题机制结果。

### 10.3 外部 benchmark

最低确认集预先固定为：

- F0 Full；
- B03 compute-matched neutral-belief；
- A01 −HBD；
- A04 −DVOI；
- A06 −Reopen；
- A08 −Persistent belief（仅限已经完成 meta-aware 顺序适配的 benchmark）。

F0、B03、A01、A04、A06 在四个 benchmark 的 healthy official protocol 上至少评测三个训练种子。A08 只在具备有效同文档 episode 的 benchmark 上加入；否则其外部 cross-task memory 结论记为“未测试”，而不是 no effect。该选择规则不依赖内部测试效果。资源充足时扩为九个核心组，不得只挑内部结果显著的消融。

每个 benchmark：

- 官方可比结果必须使用显式 fixed healthy spec；不得依赖 n_samples=1 时被 shuffle 的默认 world slot；
- 若要在公共 benchmark 上附加 synthetic stress，必须放在单独的 secondary 表中，不和官方 healthy 分数混合；
- 使用官方 scorer；
- 保存 <code>predictions.jsonl</code>、<code>official_metrics.json</code>、<code>agent_metrics.json</code>、<code>per_sample_scores.jsonl</code>；
- 用 <code>evaluation_integrity.json</code> 固定 checkpoint、commit、数据条数、tool budget 和 one-sample policy；
- scorer 或数据完整性失败使整个 benchmark run 无效；不能用项目训练 reward 代替官方结果。

## 11. 指标定义与机制报告

### 11.1 Belief

- Filter NLL；
- multiclass Brier score；
- ECE（固定 10 个 confidence bins，同时报告 adaptive-bin sensitivity）；
- session/context/shared factor accuracy；
- change detection delay；
- belief posterior entropy；
- observable-only 与 simulator-assisted 的差异。

### 11.2 Decision

- Relative oracle regret；
- positive-DVOI precision/recall；
- unnecessary probe rate；
- regret-selected node 与 random/root node 的 return 差；
- branch group 内 return spread；
- consensus calibration；
- reopen level accuracy、false reopen rate、change 后恢复调用数。

### 11.3 Auxiliary objective

- switch pair 数、通过过滤的比例、switch accuracy；
- pre-invariance pair 数、首次区分前 action JS；
- 两项 loss 的 raw value、weighted contribution、forward/backward 次数；
- auxiliary FLOPs/token 和 wall-clock 占比。

### 11.4 训练有效性

- valid questions、valid groups、K histogram；
- cross-question/world/prefix violation；
- zero-advantage group 比例；
- reward/utility/advantage 方差；
- action reward consumed count/token/abs sum；
- finite gradient、optimizer updates、checkpoint 数；
- skipped questions、infrastructure-invalid、context overflow；
- feature activation counts。

## 12. 统计分析计划

### 12.1 分析单位

- K 个 siblings 不是独立样本。
- 同一 PDF 下的 QA 也不是独立样本。
- 先在 question×world 层生成 paired score，再按 document 聚合。
- 训练 seed 是第二层随机效应。

### 12.2 主检验

对每个 A01–A08 计算：

<code>Δ_robust = Robust Macro-ANLS(F0) − Robust Macro-ANLS(ablation)</code>

以及：

<code>Δ_healthy = Healthy ANLS(F0) − Healthy ANLS(ablation)</code>

使用 10,000 次 paired crossed/product bootstrap：

1. 从共同的 paired training seeds 中有放回抽样；
2. 另从 PDF 集合有放回抽一组 document indices，并将**同一组 indices**用于所有被抽 seed 和所有方法；
3. 在每个被抽 PDF 内保留全部配对 question×world 观测；敏感性分析可再抽 question indices，但同一 indices 仍需跨方法/seed 共享；
4. 在 seed×document 的交叉乘积上计算配对均值差、普通未调整 95% percentile CI 和单侧 bootstrap p 值。

同时报告每个 seed 的点估计和 document-cluster CI、跨 seed mean±SD。PDF 与 seed 是交叉因素，不得在每个 seed 内各自抽一套不同 PDF。只有 3 个 seed 时联合 CI 应标记为初步；5 个 seed 才用于强确认性结论。

### 12.3 多重比较与判定

- 八个核心 <code>Δ_robust</code> 的单侧 bootstrap p 值使用 Holm–Bonferroni，FWER=0.05；同时报告普通未调整 95% CI，不把普通 CI 称作“Holm 校正 CI”。
- 机制次指标使用 BH-FDR，q=0.05，并标注 exploratory。
- 四个外部 benchmark 分别报告，不构造自定义总分；若做正式显著性声明，对 benchmark family 再做 Holm 校正。
- 不以“p&gt;0.05”证明两个方法等价。
- Healthy 非劣效只比较 paired-seed 的 F0 与 B03，检验 <code>H0: F0−B03 ≤ −0.01</code>，使用单侧 α=0.05，并报告 one-sided 95% lower confidence bound。

预注册解释阈值：

| 结论 | 判据 |
|---|---|
| 组件有稳健贡献 | Holm-adjusted 单侧 p&lt;0.05、普通 95% CI 下界 &gt;0，且点估计 ≥0.02 |
| Full 对 healthy 非劣 | paired F0−B03 的单侧 95% lower bound ≥−0.01 |
| 有效率改进 | ANLS 降幅不超过 0.01，tool calls 或 generated tokens 至少下降 10% |
| 未被有效测试 | Full 中该机制 exposure=0，或对应 opt-out 未达到 0 |
| 无法归因 | K、question units、world schedule、reward 或 prompt 长度同时变化 |

内部 test 只有 17 个 PDF，不能把 200 QA×8 worlds 当成 1,600 个独立样本。若 CI 较宽，应如实报告“不确定”，并依赖完整公共 benchmark 增强外部证据。

## 13. 失败、排除与重跑规则

| 情况 | 分类 | 处理 |
|---|---|---|
| <code>real_infrastructure</code> 且 <code>valid_for_rl=false</code> | 训练基础设施失败 | 原地重试一次；仍失败则用零权重 placeholder 保持共同 schedule，不补采新题；报告频率，达到阈值则整次 run 无效 |
| <code>world_injected</code> | 有效 POMDP 事件 | 必须保留 |
| 协议错误、非法工具参数、错误提前 final | 模型失败 | 不得改标 infrastructure |
| group size/cross-ID 错误、空 auxiliary、opt-out 无效 | 实现失败 | 整次 run 不产生方法结论 |
| checkpoint 缺失/错 schema、NaN、零梯度、无 update | 训练失败 | 修复后按相同 seed 重跑 |
| reward 全相同、zero advantage、无工具调用 | 未就绪或塌缩 | 不得解释成“组件无贡献” |
| 评测缺回答、工具崩溃、超预算 | ITT 模型结果 | 保留在官方分母，按 scorer 计分 |
| scorer/数据 integrity 失败 | 评测实现失败 | 整个 benchmark run 作废并修复 |

重跑规则：

- 不得删除效果差的 seed。
- 明确的基础设施故障最多自动重试一次，必须从同一 checkpoint、同一 seed、同一 world manifest 开始。
- 第二次仍失败则标记 failed，不用新 seed 替代。
- 若修复会改变算法、数据或配置，受影响的全部对照从共同 fork point 重跑，并升级实验版本号。

## 14. 后续执行 agent 的目标接口和命令模板

以下命令区分“当前已有命令”和“需要 Phase 0 新增的目标接口”。

### 14.1 数据预处理（当前已有）

~~~bash
python toolcall-rl/rl_data_preprocess.py \
  --input data/train.jsonl \
  --output data/document-qa/train.jsonl \
  --document-root data \
  --default-metric anls \
  --check-files
~~~

生成 world/meta manifest 时可以复用：

~~~bash
python toolcall-rl/build_bayestool_data.py \
  --input data/document-qa/train.jsonl \
  --output data/document-qa/train_bayestool.jsonl \
  --meta-output data/document-qa/train_bayestool_meta_raw.json \
  --seed 42
~~~

但当前 meta 输出不能直接交给 Stage D，且 builder 仍带旧 world/replica 语义；Phase 0 必须增加 schema 转换和 R4K4 参数验证。

### 14.2 Stage A（当前已有入口，正式执行需补 held-out calibration）

<code>run_bayestool_stage_a.py</code> 需要完整的 <code>rollout_interactions.json</code>，但当前训练日志只保证稀疏样本可见。Phase 0 必须先新增完整 writer，或提供下面这种从 dump-details 导出的目标接口（当前不存在）：

~~~bash
python toolcall-rl/export_bayestool_interactions.py \
  --rollout-data-dir outputs/stage_a_collection/dump_details/rollout_data \
  --output artifacts/seed_42/rollout_interactions.json \
  --require-complete
~~~

导出器必须验证 expected rollout IDs、event 数、question/document IDs 和 artifact hash，不能静默跳过无法反序列化的 shard。得到完整 artifact 后，当前已有入口为：

~~~bash
python toolcall-rl/run_bayestool_stage_a.py \
  --artifact artifacts/seed_42/rollout_interactions.json \
  --output-dir artifacts/seed_42/full \
  --epochs 1 \
  --device cpu \
  --fit-q
~~~

该 wrapper 可用于抽取 replay 和工程 smoke，但正式 risk 不能直接使用同一 replay 自校准。Full 和 no-HBD 都应使用 document-disjoint risk validation。并且 Q/risk 特征包含 belief posterior，必须从相同原始事件分别通过 Full/no-HBD filter 重新生成；不得让 no-HBD 复用 Full 的 <code>belief_snapshot</code>。no-HBD 的目标调用形态为：

~~~bash
python toolcall-rl/train_bayestool_belief.py \
  --input artifacts/seed_42/belief_replay.jsonl \
  --output artifacts/seed_42/no_hbd/belief_filter.pt \
  --without-hbd \
  --seed 42 \
  --q-replay artifacts/seed_42/no_hbd/q_replay.jsonl \
  --q-output artifacts/seed_42/no_hbd/bayes_q_head.pt \
  --risk-validation artifacts/seed_42/no_hbd/risk_calibration.jsonl \
  --risk-output artifacts/seed_42/no_hbd/answer_risk.json
~~~

Full 使用相同原始 calibration rows 和标签，但经 Full filter 重新计算 posterior features；两组的 row IDs、document IDs 和标签必须完全相同，派生 feature 文件及 hash 分别保存。Full 直接调用 trainer 时使用同样参数但省略 <code>--without-hbd</code>，并显式提供自己的 <code>--q-replay</code>。

### 14.3 训练（Phase 0 后的目标接口，当前不存在）

~~~bash
python toolcall-rl/run_qwen3_vl_4b_bayestool_ablation.py \
  --ablation-id F0 \
  --stage c \
  --seed 42 \
  --train-data manifests/train_stage_c.jsonl \
  --eval-data manifests/dev_healthy.jsonl \
  --hf-model /models/Qwen3-VL-4B-Instruct \
  --load-checkpoint checkpoints/seed_42/stage_b \
  --belief-checkpoint artifacts/seed_42/full/belief_filter.pt \
  --q-checkpoint artifacts/seed_42/full/bayes_q_head.pt \
  --risk-checkpoint artifacts/seed_42/full/answer_risk.json \
  --run-manifest manifests/experiment_v1.json \
  --output-dir outputs/ablation_v1/F0/seed_42/stage_c \
  --dump-details outputs/ablation_v1/F0/seed_42/stage_c/dump_details
~~~

launcher 内部必须将 ID 映射到固定参数，禁止从运行目录名称猜配置。A02/A03 还应显式产生：

~~~text
A02: use_switch_loss=false, switch_loss_weight=0
A03: use_pre_invariance=false, preinv_loss_weight=0
~~~

A05 映射为 <code>branch_selector=fixed_root</code>，而不是 <code>siblings=0</code>。所有组都使用 <code>group_fallback=fixed_root_k4</code>。A07 映射为 <code>meta=false, persistent=true, return=single_question</code>；A08 映射为 <code>meta=true, persistent=false, persistence_reset_scope=belief_only, return=suffix</code>。

### 14.4 训练审计（当前已有）

~~~bash
python toolcall-rl/analyze_bayestool_training_log.py \
  outputs/ablation_v1/F0/seed_42/stage_c/launcher.log \
  --output-dir outputs/ablation_v1/F0/seed_42/stage_c \
  --json-out outputs/ablation_v1/F0/seed_42/stage_c/rl_audit.json \
  --strict
~~~

必须同时运行本方案定义的 group 和 feature-activation audit，因为 strict 当前不要求 DVOI、branch、switch、pre-invariance 或 reopen 实际发生。

### 14.5 评测与官方打分（当前 wrapper 可复用，但 launcher/load 语义需先修复）

~~~bash
OPENCLAW_BAYESTOOL_LOAD=checkpoints/seed_42/final \
bash toolcall-rl/eval_benchmarks/run_eval_only.sh \
  --eval-data manifests/internal_test_healthy.jsonl \
  --output-dir results/ablation_v1/F0/seed_42/internal_healthy \
  --model /models/Qwen3-VL-4B-Instruct \
  --launcher toolcall-rl/eval_benchmarks/run_qwen3_vl_eval_launcher.py
~~~

随后导出：

~~~bash
python toolcall-rl/eval_benchmarks/export_eval_predictions.py \
  --eval-pt results/ablation_v1/F0/seed_42/internal_healthy/dump_details/rollout_data/eval_0.pt \
  --output results/ablation_v1/F0/seed_42/internal_healthy/predictions.jsonl
~~~

公共 benchmark 分别调用：

~~~text
score_docvqa2026.py
score_longdocurl.py
score_mpdocvqa.py
score_dude.py
summarize_agent_metrics.py
~~~

## 15. 每个 run 的产物契约

建议目录：

~~~text
outputs/ablation_v1/
  F0/
    seed_42/
      stage_b/
      stage_c/
      stage_d/
      eval/
  A01/
  ...
~~~

每个训练 run 必须有：

- <code>run_manifest.json</code>：Git SHA/patch hash、data/world/checkpoint hash、硬件、seed；
- <code>resolved_config.json</code>：完整解析后参数；
- <code>launcher.log</code>；
- <code>rl_audit.json</code>；
- <code>feature_activation.json</code>；
- <code>grouping_report.json</code>；
- <code>dump_details/rollout_data/*.pt</code>；
- <code>dump_details/train_data/*.pt</code>；
- checkpoints 与 SHA256；
- <code>resource_usage.json</code>：GPU-hours、tokens、wall time、峰值显存；
- <code>FAILED.json</code> 或 <code>COMPLETED.json</code>，二者只能存在一个。

总表必须能由这些文件自动重建，禁止人工复制日志中的最佳数字。

## 16. 结果表模板

### 16.1 核心结果

| ID | Seeds | Healthy ANLS | Robust Macro-ANLS | Nonstationary ANLS | Worst-world | Calls | Tokens | Infra-invalid | Δ Robust vs F0 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F0 |  |  |  |  |  |  |  |  | — |
| A01 |  |  |  |  |  |  |  |  |  |
| A02 |  |  |  |  |  |  |  |  |  |
| A03 |  |  |  |  |  |  |  |  |  |
| A04 |  |  |  |  |  |  |  |  |  |
| A05 |  |  |  |  |  |  |  |  |  |
| A06 |  |  |  |  |  |  |  |  |  |
| A07 |  |  |  |  |  |  |  |  |  |
| A08 |  |  |  |  |  |  |  |  |  |

### 16.2 机制验证

| ID | HBD metric | Switch act/loss | PreInv act/loss | DVOI selected | Regret groups | Reopen/change | Meta episodes | Belief carries | Valid groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 |  |  |  |  |  |  |  |  |  |
| 对应消融 |  |  |  |  |  |  |  |  |  |

## 17. 执行顺序与停止条件

1. 冻结 <code>experiment_v1.json</code>、数据 split 和 8-world manifest。
2. 完成 G0.1–G0.7 的实现与 deterministic tests。
3. 完成 32-question Full smoke。
4. 完成 8 个 variant manipulation smoke。
5. 完成每组 256-question one-seed engineering run。
6. 在不查看内部 test/公共 benchmark 的条件下修复所有工程问题并冻结代码。
7. 运行 F0+A01–A08 的 seeds 42/43/44。
8. 运行 B00–B03。
9. 一次性运行内部 200×8 world 评测。
10. 按预注册最低确认集运行四个公共 benchmark。
11. 若做投稿级确认，为全部九组补 seeds 45/46，再按同一分析脚本更新结果。
12. 只有全部产物、完整性检查和统计脚本通过后才生成论文表格。

以下任一条件触发停止，不进入下一阶段：

- launcher 不是 self-contained；
- 任何核心组的 group contract 不满足；
- Full 某目标机制零曝光；
- 对应消融仍有非零目标机制；
- artifact schema/hash/calibration 不匹配；
- 模型可见特征泄露 latent/gold 信息；
- reward/advantage/gradient 无有效信号；
- internal test 已被用于调参或选 checkpoint；
- 运行无法按 manifest 重建。

## 18. 解释边界

- “组件无贡献”只能在该机制被充分触发、消融真实生效且 CI 足够窄时表述；否则写“当前证据不确定”。
- Full 比 no-branch 好，但两者 K/token 不同，不能归因为 regret branching。
- Full 比 no-belief 好，但 belief prompt 长度不同，不能归因为信念正确性。
- simulator-assisted Full 的结果不能直接外推为真实工具故障鲁棒性。
- 训练 utility 改善不能替代 ANLS/Accuracy、协议成功率和实际资源成本。
- external benchmark 上的 navigation gold 信息只用于 post-hoc metric，绝不能进入 prompt、belief 或 tool state。
- 4B 与 8B 的差异是规模研究，不是组件消融。

## 19. 与当前项目文档的关系

本方案在以下项目资料基础上具体化：

- <code>README.md</code>、<code>toolcall-rl/README.md</code>；
- <code>PROJECT_DESIGN.md</code>；
- <code>docs/BayesTool-RL_方法.md</code>；
- <code>docs/BayesTool-RL_最终实现方案.md</code>；
- <code>BAYESTOOL_REQUIRED_FIX_PLAN.md</code>；
- <code>BAYESTOOL_ADVANTAGE_GROUPING_FIX_PLAN.md</code>；
- <code>BAYESTOOL_PROJECT_UPDATE.md</code>；
- <code>toolcall-rl/eval_benchmarks/README.md</code>；
- 当前代码中的 config、world、belief、decision、training、rollout、loss 和 scorer 实现。

文档由 AI 辅助进行只读代码审计和实验设计；所有数值结果仍需由固定 manifest、原始日志、官方 scorer 与统计脚本复现。
