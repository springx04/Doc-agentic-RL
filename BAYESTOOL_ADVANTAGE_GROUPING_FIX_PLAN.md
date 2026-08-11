# BayesTool-RL 统一分支与动态多题批处理实现方案

> 状态：仅实现规划，不修改代码
> 日期：2026-08-09

## 1. 最终配置边界

| 配置项 | 约束 |
|---|---|
| 优势与权重原子 | 1 道原始问题；不同题不共用 advantage baseline |
| `global_step` | 一个动态多题 global batch，对应 1 次 `optimizer.step()` |
| 每 step 题数 | 按 token/cost 动态决定；初始目标 2～4，允许 1 个超长题，第一阶段上限 8 |
| global batch 边界 | 目标训练 token/cost、题目原子性、policy version 和题数上限共同决定 |
| 默认 world slots | 4 个：healthy、local degradation、shared-family fault、change |
| 额外异常 world realizations | 0～2 个 |
| 每题基础轨迹数 `R` | 4～6 条 |
| 每个 realization 的 policy group | 恰好选择 1 个决策节点；根节点或中途节点 |
| 每个 group 的轨迹数 `K` | 4 或 8，包含 primary |
| 每题 policy records | 最少 `4×4=16`，最多 `6×8=48` |
| 每题进入训练次数 | 1 次；整题放入一个 global step，但一个 step 可包含多题 |
| 递归分支 | 第一阶段禁止 branch child 再分支 |

这里的 `K=8` 表示 `1 primary + 7 children`，不是额外再加 8 条。固定两个 replicas 的旧机制删除；初始节点 branch 已覆盖完整轨迹 replicas 的功能。

## 2. 数据结构

### 2.1 World Realizations

每题固定创建四个默认 realizations：

```text
H: healthy
L: local degradation
F: shared-family fault
C: abrupt/gradual change
```

三个异常 slots 合计最多再选择两个具体变体，而不是每个 slot 各加两个。例如 OCR 退化和 table parser 退化属于两个不同 `latent_world_id`，各自新增一条基础轨迹。

```text
R = 1 + m_local + m_shared + m_change
m_local, m_shared, m_change >= 1
m_local + m_shared + m_change <= 5
4 <= R <= 6
```

四类 world 固定不变；新增的是某一类别内的具体 realization，不是新增 world 类别。

### 2.2 统一 Decision Group

根节点和中途节点统一为 `decision_event`：

```text
decision_group_id = hash(
    objective_version,
    episode_content_id,
    question_id,
    latent_world_id,
    initial_input_hash,
    decision_event_id,
    decision_prefix_hash,
    runtime_state_digest,
    rng_coupling_id,
    return_definition_version,
)
```

同一个 group 必须满足：

- 同一题目、文档和初始多模态输入；
- 同一 `latent_world_id`；
- 同一决策前缀、工具历史、world schedule、剩余预算和 runtime 状态；
- 4 或 8 次独立 on-policy continuations；
- 相同 reward/utility、horizon 和 bootstrap 定义；
- 没有 infra error 或 context overflow；模型自身协议失败可作为真实负样本；
- 不允许从其他 world、其他节点或复制记录补齐。

初始节点 event 生成多条完整轨迹；中途节点 event 只训练分叉动作及后续 suffix，共享 prefix 必须 mask。

## 3. 多题动态组成一个 Global Step

### 3.1 先完成题目级数据与权重

每道题先独立完成：

```text
1. 使用同一 pi_old 选择并生成 4～6 个 world realizations
2. 每个 realization 运行 primary trajectory 并保存候选节点
3. 每个 realization 选择且只选择一个 decision node
4. 从该节点补采 K-1 条 siblings，使 K=4 或 K=8
5. 每个 decision group 内独立计算 advantage
6. 预计算 group→variant→slot→question 的 loss weights
7. 将整题放入对应 policy_version 的 ready-question queue
```

题目是优势和权重归一化原子，不再是 optimizer step 边界。不同题可以进入同一个 global step，但不能共用 advantage baseline。

ready item 至少携带：

```text
question_id / policy_version
16～48 条 records
每条实际 text tokens、visual tokens、有效 loss tokens
预估训练 cost
每条 group/slot/question loss weight
old_log_probs
```

### 3.2 Token/Cost-Aware Global Batch

从同一 `policy_version` 的 ready queue 中选择多个完整问题，直到达到目标训练成本：

```text
target_global_train_cost
questions_per_step_target = 2～4
questions_per_step_max = 8
```

题目不能跨 global steps 拆开；单个超长题超过 target 时允许独占一个 step。固定 `global_batch_size=8` 不再决定更新边界，应拆成 `micro_batch_size_per_rank`、`max_tokens_per_microbatch` 和 `target_global_train_cost`。

一个 global step 的流程为：

```text
1. 动态装入 Q 道完整问题
2. 按实际训练 cost 将所有 records 分配到各数据并行 ranks
3. 完成若干同步 microbatch rounds 和 gradient accumulation
4. 对整个动态 global batch 只调用一次 optimizer.step()
5. scheduler.step(); global_step += 1
```

一次遍历 1000 道有效题时：

```text
global_steps ≈ ceil(1000 / average_questions_per_step)
```

若平均每 step 为 2～4 题，则约 250～500 个 global steps。被跳过的无效题不进入 ready queue。

### 3.3 避免某张卡成为 Straggler

训练记录生成完成后长度已知，因此不用按样本条数平均分卡，而按校准后的实际 cost 分配：

```text
train_cost
  = text_loss_tokens
  + alpha * visual_tokens
  + beta * sequence_overhead
```

执行以下调度：

1. 将 records 或可打包 microbatches 按 cost 从大到小排序；
2. 使用 longest-processing-time-first，把下一个任务分给当前累计 cost 最低的 rank；
3. 相近长度/视觉 token 的序列优先做 sequence packing，减少 padding；
4. 所有 ranks 必须执行相同数量的 backward collective rounds；负载较轻的 rank 在同一 round 放更多短序列；
5. zero-loss dummy 只用于无法完全对齐的最后一个 round，并设置严格比例告警；
6. 用实际 rank step time 和 collective wait 迭代校准 `alpha/beta`，不能长期只按 token 数静态估算；
7. 极端长序列按长度 bucket 与其他长序列同轮分布，避免只让一个 rank 独占长样本。

FSDP/Megatron 的 batch planner 必须生成全局一致的 round manifest，保证 collective 次序相同；每个 rank 可以处理不同题目的 records，但所有 loss weights 必须来自打包前已经验证的题目/group metadata。

### 3.4 Rollout 侧并发

- primary、candidate 和 branch continuation 放入连续批处理队列，不按整题串行等待；
- 某条轨迹等待 OCR、解析或其他工具 I/O 时，推理 GPU 继续执行其他题/trajectory；
- branch children 使用 batched shared-prefix generation；
- ready-question queue 设置高低水位；调度同时考虑预计完成成本和等待年龄，禁止长题长期饥饿；
- 不混用超出允许 policy lag 的数据；保留 `old_log_probs`、ratio、clip 和 KL 监控。

### 3.5 吞吐自动调节顺序

1. 先用短 profile 标定每 rank 的 `max_tokens_per_microbatch`，保留约 10% 显存余量；
2. OOM 时降低单 microbatch token cap、增加 accumulation rounds，不先删 world/branch 数据；
3. rank 时间差过大时，扩大可装箱的 ready-question 候选池、重新拟合 cost model，再调整长度 buckets；
4. padding/dummy 比例高时优先改善 packing 和 round manifest；
5. GPU 利用率低且显存有余量时，提高 per-rank token cap 或每 step 目标题数；
6. policy lag、KL 或 clip fraction 超限时，降低 ready queue 年龄和每次 rollout 预生成题数，而不是降低同题 sibling 质量；
7. 每次只调整一个维度并记录 GPU-hour、reward/ANLS 与负载指标，避免吞吐提升掩盖 RL 退化。

## 4. 优势与分层 Loss

### 4.1 Group 内优势

每个 decision group 独立计算：

```text
baseline_g = sum(weight_i * utility_i) / sum(weight_i)
advantage_i = utility_i - baseline_g
```

不同题目、world realizations、branch nodes 永不共用 baseline。

### 4.2 分层权重

增加 world variants 或把 `K=4` 升为 `K=8`，用于提高估计质量和动作覆盖，不能自动增加该 world 对模型的总权重。

```text
L_group(s,v) = mean(loss_i over K siblings)
L_slot(s)    = mean(L_group(s,v) over variants in slot s)
L_question   = sum(slot_weight_s * L_slot(s))
L_step       = mean(L_question over Q packed questions)
```

默认四个 slots 的 `slot_weight_s=0.25`。如果 curriculum 改变权重，必须显式配置并记录，不能由某类 world 生成更多记录而隐式改变。

分布式 ranks 可以承担不同数量的 records，但禁止每个 rank 先求 `local mean loss` 再 all-reduce。每条记录必须在装箱前得到全局 group/slot/question 权重；各 rank 只累计 weighted loss sum，并使用同一个 global normalizer。若后端默认对 rank gradients 求平均，需要补偿 data-parallel world size，确保结果与单进程计算同一 `L_step` 数值等价。

该结构保证：

- `K=8` 不会天然获得 `K=4` 的两倍梯度权重；
- 一个 slot 增加多个 fault targets 不会挤占其他 slot；
- 一题有更多 records、另一题更短时，两题在 `L_step` 中仍然等权；
- 多个题和 microbatches 累计后，一个 global batch 只更新一次；
- world 改变动作相对优劣的信号保留，world 整体不可控难度偏移不会污染 baseline。

## 5. 多基础轨迹如何选择

### 5.1 四条必选轨迹

每题必须包含 healthy、local、shared-family、change 各一个 realization。异常参数必须在 rollout 前确定，禁止依据当题答案、答案页面、最终 reward 或失败结果事后选择。

### 5.2 额外 0～2 条轨迹

从异常候选中按以下分数选择：

```text
world_score
  = task/tool relevance
  + rolling coverage deficit
  + historical learning progress
  + historical uncertainty/calibration gap
  + recoverability and diagnostic value
  + seeded exploration bonus
  - duplicate similarity
  - estimated tool/generation cost
```

选择要求：

- 目标工具或工具族与题目可能的解题路线相关；
- 优先补足近期欠采样的 tool/fault family；
- 优先仍有可学习改善、而不是长期不可解的 fault；
- 至少保留一部分固定种子随机探索，避免被当前 Q/belief 错误锁死；
- OCR 后端缺失、工具服务崩溃等 infra 问题不得作为训练 fault；
- 记录候选集合、得分、采样概率和最终选择原因。

`R` 的最终平均值不预先写死。只允许在 4～6 内通过消融调参；不能为了增加 RL 数据默认总取 6，也不能长期只取 4 而使重要 fault targets 覆盖不足。

## 6. 每条基础轨迹如何选择分支节点

### 6.1 候选节点

每个 world realization 先运行 primary trajectory，并保存：

- 初始策略动作前的根节点；
- 有效工具观察后的动作节点；
- retry、换工具、probe、answer、reopen、abstain 等关键决策节点；
- change world 中疑似状态变化前后的节点。

每个 realization 最终只选择一个节点。没有合格中途节点时回退根节点，因此 replicas 不需要独立实现。

### 6.2 节点评分

节点只能使用策略在线可见的信息评分：

```text
branch_score
  = policy/Bayes action disagreement
  + belief entropy or change probability
  + Q uncertainty or small top-Q gap
  + expected decision regret
  + positive DVOI
  + answer/tool risk
  + action/state coverage novelty
  + remaining recoverable reward
  - continuation token cost
  - repeated-prefix penalty
```

选择得分最高且满足以下条件的节点：

- runtime checkpoint 可完整恢复；
- 剩余工具预算和 continuation horizon 足够；
- 至少可以获得 4 条独立有效 continuations；
- 不使用 hidden world label、ground-truth answer 或事后真实 reward 选节点；
- 分支后所有 siblings 继承相同的分支前 world/runtime 状态。

## 7. K=4 与 K=8 的选择和调参

### 7.1 基本规则

- `K=4` 是每个 realization 的硬下限；
- `K=8` 用于高不确定、高动作价值差异或高诊断价值节点；
- K 条记录必须是独立 on-policy samples，不能复制 token/reward；
- 不强制首个 canonical action 有 K 种，但必须监控完整轨迹去重率、动作多样性和 utility 方差；
- 完全相同且零方差的 siblings 标记为 `degenerate_no_signal`；在固定重采样上限后仍退化时，该 group 贡献零 policy loss，但其他有效 groups 仍可完成该题更新；
- 更多有效、独立且有回报差异的 siblings 通常降低优势方差并改善探索；重复、低质量或错误 bootstrap 的 siblings 只增加成本，不能假定越多越好。

### 7.2 防止 K=8 太少

第一轮调参使用以下约束：

```text
K8 target ratio: 50% of eligible decision groups
K8 rolling floor: 25% over the latest 100 valid groups
K8 hard ceiling: 100%
```

低于 rolling floor 时，提高高分候选获得 K=8 的优先级，但不能用复制或明显退化的 group 强行补配额。最终比例通过消融确定，不以 25% 或 50% 作为不可修改常数。

### 7.3 分阶段调参

在相同原始题目和尽可能相近的 generation-token 预算下进行：

1. 固定 `R=4`，比较 K8 比例约 25%、50%、75%；
2. 采用最佳 K8 策略，比较 `R=4/5/6`；
3. 在最佳区域微调 world/branch score threshold；
4. 固定数据生成配置，比较每 global step 目标 2/4/8 题及不同 token-cost budgets；
5. 用独立验证集选择配置，不使用训练 reward 直接定最终参数。

选择目标按优先级为：

```text
RL有效性：nonzero advantage、reward、ANLS、完成率、恢复能力
估计质量：utility variance、Q ranking/calibration、risk calibration
样本质量：独立轨迹率、动作多样性、协议正确率
效率：reward/ANLS gain per GPU-hour、generated tokens per valid group
```

最终配置必须同时满足最低信号和计算预算；不能只追求最少记录，也不能只追求最多 records。

## 8. Belief、Q、DVOI 与 Risk 的实现职责

| 模块 | 在线职责 | 训练职责 |
|---|---|---|
| Belief posterior | 根据可见工具观察推断工具质量、family fault 和 change state | 用合法 world/event supervision 做 filtering/calibration |
| Confidence/entropy/consensus/OOD | 判断状态与动作有多不确定 | 选择 branch 强度并报告校准 |
| Bayes Q | 估计 belief、动作和预算条件下的未来 utility | 排序动作、选择节点；用 sibling realized returns 监督 |
| DVOI | 判断额外 probe/retry 的预期收益是否超过成本 | 监督信息获取决策 |
| Risk | 决定 answer、continue、reopen 或 abstain | 用答案正确性与证据充分性校准 |

这些量既参与 world/branch 选择，也服务 agent 在线决策和辅助训练。它们不能替代 realized utility 计算普通 GRPO advantage；hidden world label 不能进入模型可见输入。

## 9. 失败与回退

1. 中途节点无法得到 4 条有效 siblings：回退根节点并有界重采样；
2. 同一 world 的 infra-invalid 轨迹：只允许在同一 `latent_world_id` 下重试；
3. 任一 realization 在根节点有界重试后仍不足 4 条有效记录：整题标记 `question_skipped`，不进入 ready queue；
4. 模型自身协议错误：保留为负样本，除非整组都无法形成可训练 token；
5. 不允许其他 world、其他题目、zero-loss dummy 或复制轨迹补足 policy group；
6. dummy 只用于分布式 shape 对齐，必须保持 zero loss 且不计入 K。
7. 如果所有 groups 都是 `degenerate_no_signal` 或题目总梯度为零，则该题不进入 global batch。

## 10. 必须记录的指标

每题输出：

```text
question_id
policy_version
world_realization_count
world_variant_count_by_slot
decision_group_count
root_vs_mid_branch_count
group_size_histogram_4_8
k8_ratio_rolling
candidate_generation_count
valid_independent_trajectory_count
trajectory_dedup_rate
canonical_action_diversity
utility_variance_by_group
nonzero_advantage_group_count
generated_tokens_by_group
question_policy_loss
kl / clip_fraction
tool_error / infra_error / protocol_error / context_overflow
belief / Q / risk losses and calibration
ready_queue_wait_ms
```

每个 global step 输出：

```text
global_step_before / global_step_after
optimizer_step_call_count
questions_in_step
question_ids_hash
policy_version / policy_lag
records_in_step
train_tokens_in_step
microbatch_round_count
records / tokens / estimated_cost per rank
rank_forward_backward_time_ms
rank_collective_wait_ms
rank_cost_imbalance_ratio
padding_token_ratio
zero_loss_dummy_ratio
GPU utilization / peak memory per rank
step_policy_loss / grad_norm
weight_sync_status
```

强制 invariant：

```text
4 <= world_realization_count <= 6
decision_group_count == world_realization_count
all policy group sizes in {4, 8}
total policy records <= 48
each valid question appears in exactly one global step
1 <= questions_in_step <= configured_questions_per_step_max
records_in_step <= 48 * questions_in_step <= 384
all questions in a step share one policy_version
step policy_lag <= configured maximum
optimizer_step_call_count == 1
global_step_after - global_step_before == 1
all ranks execute identical backward collective round count
cross_question_advantage_group_count == 0
cross_latent_world_advantage_group_count == 0
cross_prefix_advantage_group_count == 0
```

第一阶段效率目标：`max_rank_step_time / mean_rank_step_time <= 1.10`、`zero_loss_dummy_ratio <= 2%`、`padding_token_ratio <= 15%`、collective wait 不超过 step walltime 的 10%。若未达到，先调整 cost model、长度 buckets、装箱和每 step 题数，不通过减少有效 world/branch 数据掩盖调度问题。

## 11. 实现与测试清单

### M1：数据生成

- 删除固定 replicas 调度；
- 固定四个 world roots，增加 0～2 个可审计 variants；
- 每个 realization 保存根节点和中途 checkpoint；
- 每个 realization 只选择一个 branch event；
- 生成 K=4/8 并执行全部身份校验；
- 保证每题记录总数不超过 48。

### M2：优势与训练

- 每个 decision group 独立计算 advantage；
- 实现 group→variant→slot→question→global-step 分层平均；
- 支持可变数量 sequence microbatches；
- 实现按 policy version 分区的 ready-question queue；
- 实现 token/cost-aware 多题 global-batch planner；
- 实现 LPT rank 分配、长度 buckets、packing 和全局一致 round manifest；
- 同一 global step 可以跨题分配 records，但 advantage group identity 不变；
- 使用预计算全局 loss weights/sum 和 global normalizer，禁止 rank-local mean 改变目标；
- 一个动态 global batch 全部 backward 后只调用一次 `optimizer.step()`、scheduler step 和 global-step increment；
- K4/K8、extra variants 不改变题目总 loss scale；
- 保留 old log-probs、ratio、clip 和 KL 监控。

### M3：自动选择与调参

- 实现 world score、branch score 和固定种子探索；
- 实现 K8 rolling target/floor；
- 实现题目级 token/walltime budget；
- 运行 R、K8 比例和动态 global-batch cost 的分阶段消融；
- 自动标定 per-rank microbatch token cap 和 text/vision cost model；
- 比较每 step 目标 2/4/8 题及 token-cost budgets，选择每 GPU 小时收益最高且 policy lag 合格的配置；
- 将最终参数写入唯一配置入口和 run manifest。

### M4：单元与集成测试

- `R=4,K=4` 的单题得到 16 条，`R=6,K=8` 得到 48 条；
- 一个 global step 可同时包含多个不同题目，但每个 advantage group 的 `question_id` 唯一；
- 同一组题以完整 batch 与多 rank/microbatch 累计执行时，最终梯度在容差内一致；
- ranks 记录数不等时，分布式 weighted-sum 梯度仍与单进程 `L_step` 一致；
- 任意 rank 的 backward collective round 数不一致时 hard fail；
- LPT/token-cost planner 相比按记录数 round-robin 显著降低 rank step-time imbalance；
- K4/K8、长短题混合和不同题数都只触发一次 optimizer update；
- 同一题不会跨两个 global steps 重复训练；
- K8 group 与 K4 group 的总权重相同；
- 同 slot 增加 variant 不改变该 slot 总权重；
- 每题在 global-step loss 中等权，不因 records/token 更多而自然增权；
- 不同 world、prefix、题目无法混组；
- 不足 4 条时执行根节点回退，仍失败则该题不进入 ready queue；
- FSDP/Megatron 的 loss、global step 和权重同步一致。

## 12. 训练运行约束

- 同时记录 `global_step`、`questions_seen` 和 `train_tokens_seen`；optimizer/scheduler 按 global step，数据覆盖与 epoch 按 questions seen；
- checkpoint 不按每题保存；默认每处理 100 道有效题保存一次，不受动态每-step题数改变；
- 只保留最近 2 个 checkpoint 和最佳验证 checkpoint，保留策略必须在删除前验证目标路径；
- eval 与 checkpoint interval 分开配置；
- 完整训练前先运行固定种子的短测试，验证单题 16～48 条上限、多题动态 global batch、四类 world、branch、belief/Q/risk、负载均衡、loss、梯度和权重同步；
- 完整训练和最终 200 题评估使用固定 manifest，评估数据不进入 optimizer；
- 当前正在运行的旧训练不在本规划阶段停止或覆盖，修复后的新训练从干净 checkpoint 启动。
