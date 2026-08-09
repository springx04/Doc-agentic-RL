# BayesTool-RL 优势分组隔离修复方案

> 状态：仅规划，不修改代码
> 日期：2026-08-09
> 目标：确保策略优势只在同一任务内容、同一潜在工具世界和同一决策条件下计算，彻底隔离跨 world 奖励难度差异。

## 1. 结论

当前 BayesTool-RL 使用简单的组内加权平均奖励作为 baseline。在这种估计器下，不同 world 不能进入同一个策略优势组。

原因是 world 会改变工具可用性、语义准确性、结构保真度、调用成本、context rule、共享工具族故障和变化时刻。两个轨迹即使来自同一道题，也可能因为 world 难度不同而获得不同奖励；此时直接计算

```text
advantage_i = reward_i - group_mean_reward
```

会把“环境更难/更容易”误当成“动作更差/更好”。这会产生错误的策略梯度。

本项目采用以下强约束：

```text
同一个策略 advantage group
    = 同一任务内容
    + 同一 question
    + 同一 latent world 实例
    + 同一策略条件点（初始输入或完全相同的分支前缀）
```

需要特别说明：不要求分叉后的整条轨迹继续调用完全相同的工具。比较的目的正是评估不同动作；必须相同的是动作发生前的输入、可见历史和 world，动作之后的工具调用与观察可以因策略选择而不同。

跨 world 数据仍然有价值，但只能用于 world/belief/Q/switch 等显式建模环境差异的辅助目标，不能进入普通策略 reward baseline。

## 2. 当前问题与证据

既有 `BAYESTOOL_REQUIRED_FIX_PLAN.md` 已规定：

- 任意 Bayes-GRPO group 只能包含一个 `latent_world_id`；
- 正常 sibling group 只比较同一 latent world 的 replicas；
- branch group 只能比较共享同一 prefix 的分支；
- 缺少 `latent_world_id` 的旧记录只能离线诊断，不能进入新训练分组。

但当前训练路径仍优先使用宽泛的 `sibling_group_id`：

1. rollout 侧把 metadata 中的 `sibling_group_id` 直接写入 `bayes_group_ids`；
2. FSDP/Megatron 侧只按该 ID 求组内奖励均值，没有再次检查任务、world 和 prefix 是否一致；
3. branch ID 由已有 sibling ID 继续拼接 prefix，若父 ID 已跨 world，分支 ID 也可能继承错误范围；
4. 缺少强制 invariant，错误分组可以静默进入 optimizer。

已观察的 rollout 3 有 72 条有效训练记录、2 道原始题和 7 个 advantage group，group size 为：

```text
31, 4, 22, 4, 4, 3, 4
```

其中大组包含多个 `world_id` 和多种 `world_type`。这与期望的“每个 latent world 内比较 replicas”不一致，也说明 72/8 得到的 9 个 optimizer step 与 advantage group 完全不是一回事。

## 3. 分组身份模型

### 3.1 必须保留的身份字段

```text
episode_content_id
  └─ question_index / question_id
      └─ latent_world_id
          └─ replica_id
              └─ trajectory_id
                  └─ branch_event_id / decision_prefix_hash
```

字段语义：

- `episode_content_id`：原始数据样本、文档/图像内容、问题集合与顺序、prompt/tool schema 版本的稳定摘要；
- `question_id`：当前问题的稳定 ID，不能只依赖 batch 内位置；
- `latent_world_id`：完整潜在工具环境实例，包含 world type、目标工具/工具族、质量参数、session state、context rules 和 schedule；
- `replica_id`：同一 latent world 下的独立采样轨迹；
- `trajectory_id`：一条实际 rollout 的唯一 ID；
- `decision_prefix_hash`：分支动作发生前，模型全部可见输入和历史的稳定摘要；
- `branch_event_id`：同一父轨迹上的一次明确分支事件。

`world_type` 不能代替 `latent_world_id`。两个 world 即使都叫 `abrupt_change`，其目标工具、变化时刻和质量参数也可能不同。

### 3.2 初始轨迹优势组

普通完整轨迹从同一初始 prompt 开始，推荐分组键：

```text
trajectory_group_id = hash(
    objective_version,
    episode_content_id,
    question_id,
    latent_world_id,
    initial_input_hash,
)
```

该键不包含 `replica_id`、`trajectory_id` 和采样出的动作，因此同一 latent world 的多个 replicas 可以互相构成 baseline。

`initial_input_hash` 至少覆盖：

- system/user prompt 的实际 token；
- 原始问题文本；
- PDF、页面图像和其他多模态输入的内容摘要；
- 可用工具集合及 schema 版本；
- 会影响策略输入的固定配置。

### 3.3 共享前缀分支优势组

branch siblings 必须从完全相同的已观测状态分叉，推荐分组键：

```text
branch_group_id = hash(
    objective_version,
    episode_content_id,
    question_id,
    latent_world_id,
    replica_id,
    branch_event_id,
    decision_prefix_hash,
)
```

这里保留 `replica_id`，因为 branch parent 必须来自同一条实际轨迹及同一 runtime 状态。parent 与 children 共享此前全部 token、图像、工具观察、navigation state、belief snapshot、world schedule 进度和随机状态，只允许分支动作及其后续轨迹不同。

不能把不同 prefix、不同 replica 的 branch 仅因为题目相同而放在一起。

## 4. 优势计算规则

### 4.1 正常轨迹组

对同一 `trajectory_group_id` 内的有效 replicas 计算：

```text
baseline_g = sum(weight_j * utility_j) / sum(weight_j)
advantage_i = utility_i - baseline_g
```

约束：

- group size 至少为 2；
- 组内只能有一个 `episode_content_id`、`question_id`、`latent_world_id` 和 `initial_input_hash`；
- reward/utility 定义和权重版本必须完全一致；
- 保持现有“不做小组标准差归一化”的设计；
- infra-invalid 样本必须先剔除；模型自身协议失败可作为同组负样本保留；
- 有效样本不足 2 时不计算 sequence-level relative advantage，不得回退到跨 world 或跨题分组。

### 4.2 分支组

对同一 `branch_group_id` 的 parent action 与候选 branch actions 计算 sibling advantage。

除正常约束外，还必须断言：

- `decision_prefix_hash` 唯一；
- `latent_world_id` 和 `replica_id` 唯一；
- 分支前 runtime call index、schedule state 和可见工具历史完全一致；
- branch child 继承 parent world/runtime，不能重新采样 world；
- 分支之后的随机观察使用可复现的派生 seed，避免不同执行后端产生不可控差异。

### 4.3 明确禁止

- 不能按 `world_type` 聚合；
- 不能按题目 ID 单独聚合；
- 不能按宽泛 `coupling_id` 或旧 `sibling_group_id` 单独聚合；
- 不能把 optimizer batch 当成 advantage group；
- 不能在缺少关键 ID 时退回 `sample.group_index` 并继续训练；
- 不能用跨 world 的 reward normalization 掩盖环境难度差异。

## 5. Cross-world 数据的正确用途

不同 world 的样本保留，但与策略 advantage 解耦：

- belief filtering/smoothing：学习由观察推断 latent tool state；
- switch pair：同一任务内容、不同 latent world 的状态切换监督；
- pre-invariance：在证据不足时约束策略不利用隐藏 world 标签；
- BayesQ/DVOI：在明确记录 world、belief 和 action 条件后学习价值；
- robustness evaluation：分别报告 healthy、local degradation、shared fault、change world 的性能。

如果未来确实要优化“跨 world 期望效用”，必须另行定义 world-conditioned critic、分层 baseline 或重要性加权估计器，并独立证明其目标；不能继续沿用当前简单 sibling mean 并称为普通 GRPO。

## 6. 计划中的实现步骤

本节仅列出未来修改范围，本次不执行代码改动。

### M1：建立单一分组键生成器

1. 在 BayesTool 公共模块中增加规范化的 identity/group-key 构造函数；
2. normal trajectory 和 branch trajectory 使用不同 namespace；
3. 所有键带 `objective_version`，避免旧 rollout 与新规则混用；
4. 禁止 rollout、FSDP、Megatron 三处各自拼接不同格式的 key。

### M2：修正 rollout metadata

1. rollout 创建时完整写入 `episode_content_id`、`question_id`、`latent_world_id`、`replica_id`、`trajectory_id` 和 `initial_input_hash`；
2. branch checkpoint 额外保存 `branch_event_id`、`decision_prefix_hash` 和 runtime state digest；
3. branch children 继承 parent 的 latent world、replica 和 runtime state；
4. 移除宽泛 `sibling_group_id` 对策略分组的决定权，可保留其作为诊断字段。

### M3：在进入训练前做硬校验

1. rollout manager 根据规范函数生成 `bayes_group_ids`；
2. 对每组检查 content/question/world/input/prefix 唯一性；
3. 任一 cross-world group 直接失败并保存最小诊断，不允许静默训练；
4. 关键 ID 缺失的旧记录标记为 `invalid_advantage_identity`，只允许离线分析；
5. 小于 2 条的组跳过 sequence-level advantage，并记录原因。

### M4：统一两种训练后端

1. FSDP 和 Megatron 都消费 rollout manager 已验证的 canonical group ID；
2. 后端只负责数值计算，不再自行 fallback 或重建分组；
3. 两个后端对同一固定 batch 输出逐样本一致的 baseline/advantage；
4. optimizer packing 在 advantage 完成后执行，允许一个 optimizer batch 混合多个已计算好的组。

### M5：隔离辅助损失与策略损失

1. sequence policy loss 只消费 same-world advantage；
2. branch/Q/belief/switch/pre-invariance 使用各自明确的数据键；
3. cross-world auxiliary records 不得写回 `bayes_group_ids`；
4. 日志分别报告 policy advantage 与 auxiliary loss 的样本数和梯度贡献。

## 7. 必须增加的测试

### 7.1 单元测试

- 同题、同 latent world、不同 replica：normal group ID 相同；
- 同题、同 `world_type`、不同 `latent_world_id`：group ID 不同；
- 同 world、不同问题或文档：group ID 不同；
- 同 world、同题、不同初始输入/tool schema：group ID 不同；
- 同 parent prefix 的 branch：branch group ID 相同；
- 不同 prefix、不同 replica 或不同 runtime digest：branch group ID 不同；
- 缺少关键身份字段：明确拒绝，不触发 fallback。

### 7.2 数值测试

- 同组 rewards `[1.0, 0.0]` 得到 advantages `[0.5, -0.5]`；
- 加入另一个更困难 world 的样本后，原组两个 advantage 完全不变；
- 打乱样本顺序或 optimizer packing 后，逐 trajectory advantage 不变；
- FSDP 与 Megatron 的 baseline、advantage 和有效 loss mask 一致；
- singleton group 不产生伪造的跨 world advantage。

### 7.3 集成测试

以 `4 worlds × 2 replicas × 2 questions` 构造固定测试：

- normal trajectory 应形成 `4 × 2 = 8` 个 same-world/question groups；
- 每个 normal group 正常大小为 2；
- `cross_world_advantage_group_count == 0`；
- `cross_question_advantage_group_count == 0`；
- branch group 只包含同一 replica、同一 prefix 的 parent/children；
- 72 条或其他数量的最终样本可以被任意打包成 optimizer steps，但不改变上述分组。

## 8. 训练日志与验收指标

每个 rollout 必须输出：

```text
advantage_group_count
advantage_group_size_histogram
singleton_group_count
invalid_advantage_identity_count
cross_content_group_count
cross_question_group_count
cross_latent_world_group_count
cross_prefix_branch_group_count
same_world_reward_std
advantage_nonzero_count
advantage_abs_mean
advantage_group_sum_abs_max
```

验收门槛：

- 所有 `cross_*_group_count` 必须为 0；
- 所有参与 sequence policy loss 的记录都具有完整身份字段；
- normal group 仅包含同一 latent world 的 replicas；
- branch group 仅包含同一 parent runtime/prefix 的动作分支；
- 组内 advantage 加权和接近 0；
- 更换 optimizer batch size 不改变 advantage；
- 四类异常 world 与 healthy 的奖励分别统计，不再通过同组 baseline 相互抵消；
- policy loss、非零梯度和权重同步仍正常发生。

## 9. 旧 rollout 与当前训练的处理

当前训练已使用可能跨 world 的 advantage group，因此其 checkpoint 不能作为修复后“严格同 world GRPO”的可信起点。建议：

1. 保留当前日志、rollout 和 checkpoint，标记为 `cross-world-baseline diagnostic run`；
2. 不把旧 rollout 放入修复后的 replay/训练；
3. 完成代码修复后，从未受该策略梯度污染的 Stage-A/基础 checkpoint 启动新 RL run；
4. 先运行小规模固定种子验收，再启动完整 1000 题训练；
5. 当前正在运行的进程在本规划阶段不停止、不覆盖，是否终止或保留由后续实施决策处理。

## 10. 最终完成定义

只有同时满足以下条件，优势分组修复才算完成：

1. 代码层面对跨任务、跨 latent world、跨 branch prefix 分组采取 hard fail；
2. normal 和 branch 两种 group key 均有独立、稳定、版本化定义；
3. FSDP/Megatron 数值一致，packing 与 advantage 分组完全解耦；
4. 固定集成测试和真实 rollout 审计均为零 cross-world group；
5. 新训练从干净 checkpoint 启动，产生非零、可解释的 same-world advantage；
6. 最终评估按 world 分层报告，确认提升来自策略适应能力，而不是不同 world 奖励互相做 baseline。
