# BayesTool-RL 优势分组隔离修复方案

> 状态：仅规划，不修改代码
> 日期：2026-08-09
> 目标：确保策略优势只在同一任务内容、同一潜在工具世界和同一决策条件下计算，彻底隔离跨 world 奖励难度差异；同时限制 world/branch 样本膨胀，使额外推理真正提供新的训练信息。

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

- normal policy advantage group size 至少为 4；这是当前项目对稳定 GRPO 信号的最低要求；
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
5. 小于 4 条的 normal policy group 跳过 sequence-level advantage，并记录原因；不得用其他 world 补齐。

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
- 少于 4 条的 normal group 不产生伪造的跨 world advantage。

### 7.3 集成测试

以 `4 worlds × 4 replicas × 2 questions` 构造固定测试：

- normal trajectory 应形成 `4 × 2 = 8` 个 same-world/question groups；
- 每个 normal group 正常大小为 4；
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

## 10. 优势分组修复完成定义

只有同时满足以下条件，优势分组修复才算完成：

1. 代码层面对跨任务、跨 latent world、跨 branch prefix 分组采取 hard fail；
2. normal 和 branch 两种 group key 均有独立、稳定、版本化定义；
3. FSDP/Megatron 数值一致，packing 与 advantage 分组完全解耦；
4. 固定集成测试和真实 rollout 审计均为零 cross-world group；
5. 新训练从干净 checkpoint 启动，产生非零、可解释的 same-world advantage；
6. 最终评估按 world 分层报告，确认提升来自策略适应能力，而不是不同 world 奖励互相做 baseline。

## 11. World 样本是否过多的判断

结论：当前总展开量明显过多，而且增长的主要部分不是独立的新题目，而是高度相关的 branch continuations。继续硬堆数量会显著增加推理、训练步和墙钟时间，信息增益却快速递减。

### 11.1 当前展开规模

当前真实 DocVQA 配置为：

```text
每题 4 个 world slots × 每 world 2 个 replicas = 8 条基础轨迹
rollout batch = 2 道题
global batch size = 8
```

这里的 4 个 slots 已经包含 healthy，并不是“healthy + 4 个异常 world”：

```text
同一道题
├─ slot A: healthy
├─ slot B: local degradation
│          └─ single-tool 或 context degradation 二选一
├─ slot C: shared-family fault
└─ slot D: change world
           └─ abrupt 或 gradual change 二选一

每个 slot
├─ replica 0
└─ replica 1

总基础轨迹 = 4 slots × 2 replicas = 8
```

branch 发生在这 8 条基础 parent trajectories 的某个决策点之后。一次 eligible branch 当前最多增加 3 个短 continuation children；branch child 不会再各自重复生成 8 次。但是 parent 可能在多个 turn 触发 branch，因此总数仍会迅速膨胀。

branch gate 本身不按 `world_type` 限定，因此不是只有异常 world 才分支；healthy trajectory 只要满足 regret、剩余预算和 OOD 等条件也可能触发。当前观测可直接写成：

```text
36 条/题 = 8 条基础 parent + 28 条 branch children
52 条/题 = 8 条基础 parent + 44 条 branch children
```

按每个 event 最多 3 个 children 估算，分别至少需要约 10 次和 15 次成功 branch events；若某些 event 不足 3 个 children，实际 event 数还会更多。这些 event 分散在 8 条 parent trajectories 的不同 turn 上。

所以在不产生任何 branch child 时，一个 rollout 应有约 16 条基础轨迹，并产生约 2 个 optimizer steps。

实际观测：

| 观测 | 原始题数 | 最终训练记录 | 每题记录 | 相对 8 条基础轨迹 | GBS=8 的 optimizer steps |
|---|---:|---:|---:|---:|---:|
| rollout 3 | 2 | 72 | 36 | 4.5 倍 | 9 |
| 早期 rollout | 2 | 104 | 52 | 6.5 倍 | 13 |

也就是说，2 道题本应产生的 16 条基础轨迹，被展开成了 72～104 条训练记录；每个原始问题实际贡献了 36～52 条高度相关样本。

若把这一比例机械外推到 1000 道题，在全部有效且 GBS 保持 8 的情况下，大致会形成：

```text
36,000～52,000 条最终训练记录
4,500～6,500 个 optimizer steps
```

这还没有完整计入候选动作采样的推理成本，因此真实 generation 成本会更高。

### 11.2 当前隐藏的推理放大器

真实数据入口当前默认：

```text
branch_probability = 1.0
decision_regret_threshold = -1.0
max_action_candidates = 4
max_siblings = 4
branch_horizon = 3
```

这与库/Stage-C 常规默认的 `branch_probability=0.25` 不同。负 regret 阈值和 100% gate 使绝大多数满足基本条件的决策都可能分支。

此外，当前候选采样为了得到最多 4 个不重复动作，单个决策最多尝试 `4 × 3 - 1 = 11` 次额外生成；这些请求即使最终没有触发 branch，也已经消耗推理时间。一个 parent trajectory 还可能在多个 turn 分支，每次最多产生 3 个 children。

因此成本有三层，必须分别统计：

```text
基础 world/replica 轨迹
  + 候选动作额外解码（未必成为训练样本）
  + branch continuation（成为额外训练记录）
```

只看最终 `.pt` 中的样本条数，会低估真实推理次数和生成 token 数。

### 11.3 为什么继续扩充收益很低

- 同一道题的 world replicas 共享文档和任务，样本相关性远高于新增原始问题；
- branch children 共享长前缀，只在后部少数动作上不同，不能等价为完整独立样本；
- 容易触发 branch 的长轨迹会在训练中被过度加权，改变原始题目分布；
- 更多 optimizer steps 主要来自记录数量增加，不代表获得了等比例的新任务信息；
- protocol error 或低奖励轨迹若大量复制，会让某类失败模式支配梯度；
- 全部题目都做完整 world 笛卡尔积，没有充分利用“跨题轮换 world 也能覆盖总体分布”这一事实。

## 12. 推荐的训练样本预算

### 12.1 总体目标

任何参与 normal GRPO advantage 的 latent world 必须至少有 4 条完整、同 world、同初始输入的 on-policy trajectories。平均最终可训练记录仍控制在每题 5～6 条；少量 paired/coverage anchors 单独计量，不与普通题共用单题上限。

对 1000 道题、GBS=8，目标规模约为：

```text
约 5,000 条基础策略轨迹
约 5,000～6,000 条最终策略训练记录
约 625～750 个 optimizer steps
```

这比当前外推的 4,500～6,500 steps 低约一个数量级，同时仍保留 same-world relative advantage 和全部 world 类型覆盖。

### 12.2 推荐的混合采样结构

保留包括 healthy 在内的 4 个 world slots，但不再让每道题都运行全部 4 个 world。改成以下分层预算：

| 题目比例 | 采样结构 | 每题基础轨迹 | 主要用途 |
|---|---|---:|---|
| 85% 常规题 | 1 个 latent world × 4 replicas | 4 | 稳定的四样本 same-world GRPO advantage |
| 10% paired anchors | healthy + 1 个目标异常 world，各 4 replicas | 8 | 同题环境对照、switch/belief 辅助监督 |
| 5% coverage anchors | 4 个 world slots × 4 replicas | 16 | 检查全部 world 覆盖和系统性退化 |

平均基础轨迹为：

```text
0.85 × 4 + 0.10 × 8 + 0.05 × 16 = 5.0 条/题
```

这比当前每题固定 8 条基础轨迹更少，并确保每一个实际计算 advantage 的 world group 都有 4 条轨迹。

所有启用 world 类型通过跨题分层轮换保证覆盖。相同题目是否进入 anchor 集由固定种子决定，避免训练过程中随意改变对照集。

若第一阶段希望降低实现复杂度，可先采用简化方案：

```text
每道题从 4 个 world slots 中确定性选择 1 个
该 latent world 生成 4 条独立 on-policy trajectories
跨题均衡轮换 healthy/local/shared/change
合计 1 world × 4 trajectories = 4 条/题
```

这已经把基础推理减半，并满足四样本 GRPO。其缺点是普通题没有同题跨 world 对照，因此必须由 paired/coverage anchors 补充。

如果坚持每一道题都运行全部 4 个 world，同时每个 world 至少 4 条轨迹，那么最低成本必然是：

```text
4 worlds × 4 trajectories = 16 条基础轨迹/题
```

这个乘积无法通过改 batch size 或分组键消除。若既要四样本优势又要控制成本，必须采用“普通题单 world 轮换 + 少量多 world anchors”的结构。

### 12.3 World 调度原则

1. world 按全局 rolling window 分层覆盖，而不是每题做完整笛卡尔积；
2. 每 100 道题检查各 world slot 的目标份额，欠采样类型优先补齐；
3. healthy 保留稳定比例作为性能锚点，但不与异常 world 共同计算 advantage；
4. 后续可根据“学习进展”调整采样权重：奖励长期不变的简单 world 降采样，仍有可学习梯度的 world 升采样；
5. 自适应采样必须记录采样概率，评估仍使用固定 world 分布，防止 curriculum 指标失真；
6. 不能仅按最低奖励过采样，否则会反复堆积模型当前无法解决或基础设施异常的样本。

### 12.4 四条同 World 轨迹如何提高回答质量

四条轨迹必须是真正独立的 on-policy completions，而不是复制 token、复制 reward，或用不同 world 的轨迹凑数。推荐：

1. 四条轨迹共享完全相同的题目、文档、初始 prompt、tool schema 和 latent world；
2. 使用四个可复现但不同的 policy sampling seeds，并在一个 shared-prefix batched request 中生成；
3. 给完整轨迹保留足够的工具预算，优先获得完整证据链，而不是把预算花在大量短 branch 上；
4. infra-invalid 轨迹只允许由同题、同 latent world 的新轨迹替换；
5. 模型自身 protocol error 保留为负样本，但若四条全部以同一种协议错误结束，则该组记为 `degenerate_no_signal`，不强行做虚假相对优势；
6. 若四条动作/奖励完全相同，可在固定上限内补采最多 2 条同 world trajectories；仍无差异则跳过本组 policy update，并进入协议或 curriculum 诊断；
7. 正确答案、证据定位和工具纪律共同进入 utility，确保优势偏向“答对且证据可靠”的轨迹；
8. teacher/corrected trajectory 可以用于独立 SFT 或 replay supervision，但不能混入四条 on-policy GRPO baseline。

高质量的关键不是把 branch child 数量加到 4，而是让四条完整轨迹在相同环境下进行真实策略探索，并把节省的推理预算用于更完整的工具调用和证据获取。

## 13. Branch 的高信息、低数量使用方案

### 13.1 两阶段触发

在生成额外候选前先做廉价 pre-gate。只有同时满足以下条件才进入候选/branch 阶段：

- 至少已有一次有效工具观察；
- belief entropy、change probability、policy/Bayes disagreement 或 DVOI proxy 超过阈值；
- 剩余工具预算足够；
- 不是 infra error、context overflow 或不可恢复协议错误；
- 当前题尚未用完 branch 配额。

pre-gate 未通过时只执行 primary policy action，不再为了最终不会使用的 branch 生成额外候选。

### 13.2 严格数量上限

branch 必须区分两种用途：

- `policy_branch_group`：参与相对策略优势，必须由同一 prefix/world 的 `1 parent + 3 children = 4` 条轨迹构成；
- `q_only_pair`：只有 parent/alternative 两条时，只用于 Q/regret auxiliary target，不进入 GRPO baseline。

完整训练的推荐初始预算：

```text
policy-branch questions <= 全部问题的 5%
max policy branch events per selected question = 1
policy branch group = 1 parent + 3 children
branch_horizon = 1～2
paired/coverage anchors: policy branch disabled
```

这样每 100 道题最多只有 5 个 policy branch events、15 个 children，平均仅增加 `0.15` 条 child/题。它满足四样本 action comparison，又不会回到当前多 turn、全量分支的规模。

不再默认 `probability=1.0`，也不允许同一 parent 在多个 turn 重复展开。branch 配额以全局题目比例和每题 event 数为硬上限；概率或信息分数只负责决定哪些题获得这笔预算。

### 13.3 候选动作生成优化

1. primary action 始终来自当前策略；
2. 优先从合法工具 schema、navigation state 和 Q-head 候选中构造 canonical alternatives；
3. `policy_branch_group` 所需的 3 个 alternatives 使用共享 prefix 的单次 batched generation，不能最多 11 次串行尝试；
4. 若无法得到 4 个有效、具有不同 canonical action 的 siblings，则不计算 branch GRPO；已有 pair 可降级为 `q_only_pair`；
5. 不为了凑满 4 个动作无限重采样，单个 event 只有一次有上限的候选批次；
6. 对 branch policy loss 只训练分叉后的有效 suffix，共享 prefix 不重复贡献 loss。

### 13.4 Branch 数据如何发挥更大作用

- 一个合格的四轨迹 policy branch group 同时服务于 sibling action advantage、Q target 和 regret audit；
- 只有两条的 branch pair 仍可服务 Q/regret，但不得冒充四样本 GRPO；
- Q/belief 可保存紧凑 feature/replay record，不必把同一长序列重复加入策略 batch；
- policy branch sample 保持 on-policy 并立即使用，但每个 rollout 有固定 quota；
- Q/belief replay 可以跨 rollout 重采样，避免为辅助头重复运行昂贵的 VLM generation；
- 优先保留动作不同且 utility 有差异的分支；完全相同动作、相同观察或零信息分支只记录诊断，不进入策略 loss。

## 14. 更好利用 World 的计算结果

### 14.1 共享干净工具结果

对于相同 document、tool 和 arguments，先缓存一次 clean tool result，再由各 world runtime 对其施加不同 observation transform。这样可以减少重复 OCR、解析、渲染和表格抽取，但必须保持每个 world 的 corruption/event metadata 独立。

缓存只复用确定性的基础工具结果，不能复用已经被某个 world 腐化后的观察。

### 14.2 将完整 world 对照集中到 anchors

完整四 world 对照的价值主要是：

- 验证 healthy 基线没有被污染；
- 测量单工具、context、shared-family 和 change world 的相对影响；
- 构造 belief switch/pre-invariance 数据；
- 检查策略是否真正根据观察适应，而不是记住 world 标签。

这些目标不要求所有 1000 道题都运行全部 world。固定 5%～20% 的 anchors 做完整或成对对照，其余题按 world 配额轮换，可用更少推理获得更高的原始题目多样性。

### 14.3 评估与训练分离

- 训练期间的小评估使用无 branch 的固定分层子集；
- 最终 200 题评估使用固定 world 分布，并额外保留少量 all-world anchors；
- eval 样本不进入 replay 或 optimizer；
- 分别报告每个 world 的 reward/ANLS/completion/tool success，不以展开后的样本数做加权总分；
- paired anchor 报告相同题目从 healthy 到异常 world 的性能下降和恢复能力。

## 15. 资源监控与硬验收门槛

每个 rollout 新增以下计数，并按原始题目归一化：

```text
original_question_count
base_trajectory_count
candidate_generation_request_count
candidate_generated_token_count
branch_eligible_count
branch_triggered_count
branch_event_count
branch_child_count
branch_generated_token_count
final_policy_sample_count
zero_loss_dummy_count
sample_expansion_ratio
generated_tokens_per_question
optimizer_steps_per_question
walltime_per_question
world_type_coverage
```

完整训练启动前必须满足：

- 平均 `base_trajectory_count/original_question_count <= 5.0`，允许的数值误差上限为 5.2；
- 平均 `final_policy_sample_count/original_question_count <= 5.5`；
- 常规题基础轨迹固定为 4，参与 normal advantage 的每个 group size 不小于 4；
- 常规题最终 policy samples 不超过 7，paired anchor 不超过 8，coverage anchor 不超过 16；
- policy branch event 覆盖率不超过问题数的 5%，每个选中问题只允许一个四轨迹 branch group；
- 每个 policy branch event 固定为 primary parent 加 3 个 children，branch children 总量不超过基础轨迹的约 3%；
- paired/coverage anchors 禁止 policy branch，保证对照公平并固定成本；
- 每个触发的 branch 只允许一次 shared-prefix batched candidate request，不允许串行补满；
- `cross_latent_world_group_count == 0`；
- 改变 optimizer batch size 不改变样本选择和 advantage；
- 各 world 在 rolling window 内达到预定覆盖率；
- 相比无 world/branch 基线，额外生成 token、墙钟时间和奖励增益均被单独报告。

如果某项超限，rollout manager 应先停止继续扩展该题，而不是生成后再靠丢弃样本解决。生成后丢弃虽然能减少训练记录，却不能挽回已经消耗的推理时间。

最终采用哪个预算，应由一个小规模消融实验决定：比较 `8 base（4 worlds × 2）+ aggressive branch`、`4 same-world base + sparse four-sibling branch`、`4 same-world base + no branch` 三组，在相同 generation-token 预算下评估 reward、ANLS、advantage 非零率和每 GPU 小时收益。推荐默认采用 `4 same-world base + sparse four-sibling branch`，只有它相对 no-branch 产生稳定收益时才保留 branch policy loss。
