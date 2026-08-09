# BayesTool-RL 同 World 统一分支优势与采样预算方案

> 状态：仅规划，不修改代码
> 日期：2026-08-09
> 目标：把根部采样和中途分支统一为同一种 shared-prefix sibling rollout；保证策略优势只在同一题目、同一 latent world、同一决策节点内计算，同时合理利用 4 类 world、belief/Q/risk 和分支预算。

## 1. 最终结论

本方案采用一个统一原则：

```text
一个策略 advantage group
    = 同一题目与文档
    + 同一 latent world 实例
    + 同一初始输入
    + 同一决策前缀与 runtime 状态
    + 多个不同的 on-policy action/continuation
```

因此：

1. `replica` 不再作为独立的策略训练概念；
2. 第一次策略动作之前的初始节点也是一个 branch node；
3. 在初始节点生成 `1 primary + K-1 siblings`，就等价于过去所谓的 `K replicas`；
4. 在工具调用后的节点生成 siblings，则是局部、共享前缀的分支比较；
5. 两者统一使用 `decision_group_id` 和同一套优势计算；
6. `trajectory_id` 仍保留用于追踪一条实际执行路径，但不决定谁与谁计算优势；
7. 默认 group size 为 4；高信息节点可使用 8，例如 `1 primary + 7 children`；
8. 不同 world、不同 world realization、不同分支位置的记录绝不能为了凑数量放进同一个优势组。

这在结构上与 shared-prefix、ARPO-like 的局部 rollout 思路相近。branch child 可以视为该决策节点下的一次 GRPO rollout，但不是一个新的原始题目。

当前观测到的每题 36～52 条最终记录不必直接判定为异常。它可能由多个有效 branch events 产生；真正要检查的是这些记录是否组成了足够多的、严格同节点同 world、continuation 有真实差异且回报有信息的 sibling groups，以及它们消耗了多少生成 token。

## 2. 为什么不再保留独立 Replica 概念

### 2.1 过去两条 Replicas 能做什么

同一 world 下的两条完整 replicas 并不是字面复制，而是从初始 prompt 独立采样两次完整策略路径。它们可以：

- 提供完整 episode return 的两个样本；
- 覆盖不同工具路径；
- 在没有中途 branch 时产生一个非常小的相对 baseline；
- 增加可供后续分支的父轨迹节点。

但固定 `2 replicas/world` 有三个问题：

1. 两条不足以满足本项目已验证的“至少 4 条才有稳定相对优势”要求；
2. 两条从根部开始的完整轨迹会同时混入很多不同决策，credit assignment 较粗；
3. 初始节点 branch 本身已经可以生成 4 或 8 条完整路径，再额外定义 replicas 是重复机制。

### 2.2 统一后的等价关系

```text
旧表示：同一 world × 4 replicas

新表示：同一 world 的初始 decision node
        ├─ primary continuation
        ├─ sibling 1
        ├─ sibling 2
        └─ sibling 3
```

两者都从相同题目、输入和 world 状态开始。新表示的好处是：根节点和中间节点使用完全相同的分组、日志、预算与损失规则，不需要维护两套概念。

如果一个 world root 没有任何节点被选中扩展为至少 4 个 siblings，则该根轨迹可以用于 reward、belief、Q、risk、world coverage 和评估，但不强行产生相对策略优势。不能用另一个 world 的轨迹补齐。

## 3. 四类 World 与 Latent World Realization

每道题默认保留 4 个 world slots，已经包含 healthy：

```text
同一道题
├─ H: healthy
├─ L: local degradation
│     └─ single-tool degradation 或 context degradation
├─ F: shared-family fault
└─ C: change world
      └─ abrupt change 或 gradual change
```

每个 slot 默认选择 1 个具体 `latent_world_id`，所以最初只有 4 条 primary root trajectories，而不是 `4 worlds × 2 replicas = 8` 条。

必须区分：

- `world_slot`：H/L/F/C 这一大类；
- `latent_world_id`：本次实际环境实例，包括目标工具/工具族、退化参数、context rule、变化时刻、schedule、session state 和环境随机状态；
- `trajectory_id`：在该环境实例中执行的一条路径。

例如，“降低 OCR 工具可用性”和“降低表格解析工具可用性”即使都属于 local degradation，也应是两个不同的 `latent_world_id`，各自新增一条 root，分别产生自己的 branch groups。它们不能互相计算优势。

## 4. 统一的决策分组身份

### 4.1 必须保存的身份层级

```text
episode_content_id
  └─ question_id
      └─ world_slot
          └─ latent_world_id
              └─ trajectory_id
                  └─ decision_event_id
                      └─ decision_prefix_hash / runtime_state_digest
```

字段含义：

- `episode_content_id`：文档/图像、题目集合、prompt 与 tool schema 版本的稳定摘要；
- `question_id`：当前问题的稳定 ID；
- `latent_world_id`：完整的实际 world realization；
- `trajectory_id`：一条实际执行路径的唯一 ID；
- `decision_event_id`：一次明确的 sibling expansion；初始节点也有自己的 event ID；
- `decision_prefix_hash`：动作发生前模型全部可见 token、图像和历史的摘要；
- `runtime_state_digest`：工具调用序号、world schedule 进度、navigation state、预算和可见工具历史的摘要；
- `rng_coupling_id`：保证 siblings 的环境随机性以可比方式耦合。

`world_type` 或 `world_slot` 不能替代 `latent_world_id`。两个同为 abrupt change 的环境，如果目标工具或变化时刻不同，就不是同一个优势组。

### 4.2 唯一的策略分组键

根部和中途节点统一使用：

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
    return_definition_version,
)
```

这个键不包含采样出的 action，也不包含 child trajectory ID，因此同一节点的不同 action 可以进入同一组。

初始节点的 `decision_prefix_hash` 就是实际初始多模态输入和 tool schema；中间节点则覆盖此前全部可见历史。不同节点即使来自同一题、同一 world，也必须有不同 group ID。

## 5. 策略优势如何计算

### 5.1 Branch Event 是唯一的 Policy Advantage 单元

对一个有效 sibling group：

```text
baseline_g = sum(weight_i * utility_i) / sum(weight_i)
advantage_i = utility_i - baseline_g
```

一个 group 必须同时满足：

- group size 至少为 4；
- 默认 `K=4`，高价值节点可用 `K=8`；
- 只有一个 question、document、initial input 和 latent world；
- 只有一个 decision prefix 和 runtime state；
- primary 与 children 使用相同的 return 定义、branch horizon 和 reward 版本；
- 分支前的环境随机状态一致，分支后的随机性采用可复现的 coupled seeds；
- 至少有 4 个有效 on-policy continuations；
- K 条记录必须来自 K 次独立 on-policy sampling，不能直接复制 token/reward；
- 优先获得不同 canonical actions；若首个 action 相同但后续 continuation 独立分化，仍是有效策略样本；
- 若所有记录的 token、动作路径和 utility 都完全相同，则标记为 `degenerate_no_signal`，不靠复制数量制造更新；
- infra error、context overflow 等基础设施无效样本先剔除；模型自身的真实协议失败可以作为负样本保留。

`1 primary + 7 children = 8` 是有效的 GRPO sibling group，前提是这 8 条都从同一个决策节点出发。来自两个不同工具调用节点的 4+4 条轨迹是两个 group，不是一个 8 条 group。

### 5.2 根部与中途 Branch 的区别

- 根部 branch：从第一次策略动作前展开，得到多条完整 episode trajectories；它完全覆盖旧 replicas 的用途；
- 中途 branch：共享已经发生的工具观察，只比较当前动作及后续 continuation，credit assignment 更精确；
- 一个 primary trajectory 可以在多个不同节点产生 branch events，但每个 event 单独计算优势；
- 初期不建议让 branch children 再递归产生 children，以免形成指数增长；后续只有在消融证明有收益时再开放受限递归。

### 5.3 Policy Loss 应作用在哪些 Token

中途 branch 的共享前缀不能因被复制多次而重复贡献策略梯度。每个 event 的 advantage 应绑定到：

1. 该 event 选择的 action token；以及
2. 需要训练时，该 action 后续的有效 continuation suffix。

共享前缀必须 mask。若同一 primary trajectory 上有多个 branch events，应按各自 token span 写入 event-level advantage，不能复制一整条 parent sequence 并给它多个互相冲突的全序列 advantage。

根部 branch 没有策略生成的共享前缀，因此可以把每条完整 continuation 作为该根部 event 的 rollout。

### 5.4 明确禁止

- 不同 latent world 放进同一 advantage group；
- 不同题目、文档或初始输入放进同一 group；
- 不同 branch locations 仅因题目相同而合并；
- 用 optimizer batch 边界定义 advantage group；
- 用旧的宽泛 `sibling_group_id`、`coupling_id` 或 batch index 静默 fallback；
- 小于 4 条时用其他 world、其他节点或复制记录凑数；
- 把 teacher/corrected trajectory 混入 on-policy GRPO baseline。

optimizer batch 可以同时打包多个已经独立算好 advantage 的 groups。混合打包只影响吞吐，不改变优势比较对象。

## 6. 36～52 条记录为什么可能合理

设四个 slots 中实际选择的 latent world realizations 数分别为：

```text
m_H, m_L, m_F, m_C
```

通常 `m_H=1`，三个异常 slot 默认也各为 1；如果 local world 同时选择 OCR 和 table 两种退化目标，则 `m_L=2`。

总 root 数为：

```text
R = m_H + m_L + m_F + m_C
```

对第 `e` 个 branch event，令 `K_e` 是该 event 的 sibling group size，已经包含 primary。每个 event 只新增 `K_e-1` 条 continuation records，因此：

```text
N_final = R + Σ_e (K_e - 1)
```

这比 `n1 + w1*n2 + ...` 更精确，因为它把“新增 world realization”和“同一 world 内新增 branch siblings”分开了。

例子：

```text
4 个默认 world roots
+ 每个 root 在初始节点形成 8-sibling group
= 4 + 4 × (8 - 1)
= 32 条最终记录
```

再给一个异常 slot 增加 1 个 world variant，并在它上面形成 4-sibling group：

```text
32 + 1 root + 3 children = 36 条
```

52 条也可以由合法的多个 4/8-sibling events 组成。例如 6 个 world realizations、4 个 8-sibling events 和 6 个 4-sibling events：

```text
6 + 4 × 7 + 6 × 3 = 52
```

所以 36～52 不是仅凭数量就能判定错误。不过 52 条意味着该题获得了很多 branch 预算，必须证明这些 event 都有信息价值；它不应在 `branch_probability=1.0`、负 regret threshold 下无差别成为每题默认值。

还必须把最终 records 与真实 inference cost 分开统计。为了得到 4/8 个不重复 action，模型可能生成并丢弃额外候选；这些请求不出现在 `.pt` 的最终记录数里，却仍消耗 GPU 时间和 token。

## 7. 异常 World Realization 如何选择

### 7.1 选择原则

world 必须在策略执行前确定，不能根据正确答案、答案页面、最终 reward 或某条轨迹失败后再事后挑选。候选只使用公开任务输入、tool schema、历史覆盖统计和过去 rollout 的聚合学习信号。

每个候选 world variant 可使用下列分数：

```text
world_score
  = α × task/tool relevance
  + β × rolling coverage deficit
  + γ × historical learning progress
  + δ × model/belief uncertainty on this fault family
  + η × recoverability and diagnostic value
  + ε × seeded exploration bonus
  - λ × duplicate similarity
  - μ × estimated generation/tool cost
```

推荐流程：

1. healthy 固定 1 个 realization；
2. local slot 先从当前任务可能使用的工具或 context route 中选择 1 个目标；
3. shared-family slot 选择与任务相关、且 rolling coverage 不足的工具族；
4. change slot 预先选择目标与 change schedule，保证轨迹中有机会看到 change 前后的动作；
5. 每个异常 slot 默认只选 1 个 realization；
6. 只有当另一个目标同时具有较高相关性、覆盖缺口和可学习性，并且仍有 token 预算时，才增加第二个 realization；
7. 初始实现建议每题最多增加 2 个额外异常 realizations，之后根据消融结果调整；
8. 保留一小部分固定种子的探索配额，防止 Q/belief 只选择当前已经熟悉的 fault；
9. 记录候选集合、选择分数、采样概率和原因，便于审计 curriculum 偏差。

不同 degraded tool、family 或 schedule 是不同 worlds，只能各自在自己的分支节点内计算优势。它们之间的同题对照用于 robustness、belief、switch 和评估，而不是普通 GRPO baseline。

### 7.2 防止无效 World 堆叠

以下情况不应继续增加 realization：

- 目标工具与题目路线明显无关；
- 该 fault 使任务几乎必然不可解，长期没有学习进展；
- 失败来自 OCR 后端缺失、服务崩溃、协议适配错误等基础设施问题；
- 新 variant 与已选 variant 的可见效果几乎相同；
- 只是因为该 world reward 低就持续过采样；
- 已达到题目级 generated-token 或 walltime 预算。

## 8. Branch 节点与 Group Size 如何选择

### 8.1 先运行 Primary，再选择值得展开的节点

对每个 world realization 先执行 1 条 primary path，同时保存可恢复 checkpoint。初始节点天然也是候选 checkpoint。沿途节点只保存允许在策略时可见的信息，不能使用 ground-truth answer 参与选择。

候选节点至少满足：

- 有可恢复的完整模型、工具、navigation、world schedule 和 RNG 状态；
- 剩余工具预算和 continuation horizon 足够；
- 没有 infra error、context overflow 或不可恢复协议错误；
- 能从当前策略得到至少 4 个有效 on-policy continuations；
- 优先存在多个不同 canonical actions，或至少能在后续 continuation 中形成真实策略分化；
- 不会因已知确定性约束导致所有 siblings 完全相同。

### 8.2 节点信息分数

```text
branch_priority
  = α × policy/Bayes action disagreement
  + β × belief posterior entropy or change probability
  + γ × Q predictive uncertainty
  + δ × expected decision regret
  + η × positive DVOI
  + κ × state/action coverage novelty
  + ρ × remaining recoverable reward
  - λ × estimated continuation token cost
  - μ × duplicate-prefix overrepresentation
```

从候选节点中按题目级预算选择 top events，而不是对每个 turn 独立使用 `probability=1.0`。初始节点可在以下情况优先：

- 希望比较完整的不同工具路线；
- 当前 world 在近期缺少有效 policy groups；
- 中途没有任何合格节点；
- 历史上该题型的首个工具选择不稳定。

中途节点可在以下情况优先：

- 已有真实工具观察后，模型对工具可靠性判断不确定；
- 是否重试、换工具、探测、停止或作答是关键决策；
- Q top actions 接近或 policy 与 Bayes action 不一致；
- change world 中出现疑似状态变化。

### 8.3 K=4 与 K=8

- `K=4`：默认最低成本配置，满足至少 4 条的有效优势要求；
- `K=8`：只给最高信息节点、动作空间确实丰富或不确定性很高的 event；
- 候选应由当前策略以可记录 log-prob 的方式采样，使用不同可复现 seeds；
- 合法 schema 约束可以过滤无效格式，但不能把 teacher action 冒充 on-policy sibling；
- 若一次有上限的 batched sampling 后仍不足 4 个有效 continuations，该 event 只用于 Q/risk 诊断，不进入 policy advantage；
- 优先给 continuation 足够工具预算，以获得完整证据链，而不是只生成大量极短、同质化 children。

“至少 4 条”指至少 4 次独立的 on-policy continuation，并不要求第一个 canonical action 必须恰好有 4 种。强行凑 4 种首动作可能把人工候选变成 off-policy 数据。相同首动作但后续采样路径不同的轨迹仍可用于 suffix-level GRPO；只是 action diversity、完整轨迹去重率和 utility 方差必须单独监控。完全复制的记录不提供新信息。

### 8.4 Return 的可比性

最理想的 branch sibling utility 是运行到自然终止后的真实 task/evidence/tool utility。

如果为了成本只运行固定短 horizon，则所有 siblings 必须使用相同 horizon，并可用一个冻结或慢更新的 target-Q 做一致 bootstrap。此类 `bootstrapped_branch_advantage` 必须单独标记和报告，不能与 terminal-return group 混在一起；在可承受时，策略优势优先使用 realized terminal return，减少 Q 估计偏差自我强化。

## 9. Belief、置信度、Q、DVOI 与 Risk 分别做什么

它们不是用来把不同 world 的 reward 强行拉到同一尺度，而是帮助 agent 在看不到真实 world 标签时做决策，并把分支预算放在最有信息的位置。

```text
可见工具观察
    ↓
belief：推断当前可能处于什么工具/world 状态
    ↓
Q：估计在这些可能状态下，各候选动作的未来效用
    ↓
confidence / DVOI / risk：判断是否可信、是否值得再探测、是否该继续或作答
    ↓
选择 primary action 与值得展开的 branch node
    ↓
siblings 的真实回报
    ├─ 形成同节点 GRPO advantage
    └─ 反过来监督 Q、belief calibration 与 risk
```

| 量 | 含义 | 直接用途 | 不应该做什么 |
|---|---|---|---|
| Belief posterior | 根据可见历史，对 hidden tool availability、semantic accuracy、structure fidelity、family fault、regime/change state 的概率分布 | 判断哪个工具可能失效、是否发生共享故障或状态变化 | 不能把模拟器真实 world 标签直接暴露给策略 |
| Posterior entropy | 对当前 world/tool 状态有多不确定 | 高熵节点优先探测或分支 | 不是最终 reward |
| Action consensus | 在多个 posterior world particles 下，有多少比例选择同一动作 | 判断动作是否稳健；低 consensus 时增加 branch 价值 | 不能替代真实 sibling return |
| OOD score | 当前 belief/state 是否超出训练支持范围 | 降低盲目信心、触发保守策略或额外探测 | 不能把所有低奖励都解释为 OOD |
| Bayes Q(b,a) | 给定 belief、候选动作和剩余预算时的期望未来 utility，并可输出方差 | 动作排序、branch 节点选择、短 horizon bootstrap、Q auxiliary training | 不作为跨 world GRPO baseline；也不等同于 PPO critic |
| DVOI | 再调用一次诊断工具带来的期望决策收益减去成本 | 决定是否值得 probe/retry/换工具 | DVOI≤0 时不应无限调用工具 |
| Answer risk | 当前答案缺证据或错误的校准概率 | answer、continue、reopen、abstain 的门控 | 不能只看语言模型口头自信 |
| Realized utility | siblings 实际得到的任务、证据、工具纪律与成本综合回报 | 计算同节点策略 advantage，并监督 Q/risk | 不能跨 latent world 直接求简单均值 baseline |

### 9.1 Q 的训练与防止自我强化

Q 主要回答：“在当前 belief 和预算下，选这个工具动作后，预计最终能得到多少 utility？”branch siblings 提供同一状态下多个动作的 realized targets，因此是很好的 Q 监督。

但如果永远只在 Q 认为有价值的节点分支，Q 的早期错误可能让系统看不到其他动作。为此需要：

- 固定比例的 seeded random exploration events；
- 记录 selection propensity；
- 使用慢更新/target Q 做 bootstrap；
- 分别报告 Q calibration、ranking accuracy 和真实 return；
- 定期在固定节点集上做不依赖 Q 的 4/8-way branch audit；
- policy advantage 以 realized sibling utility 为主，不直接把 Q 预测当作真值。

hidden world label 可以离线监督 belief/Q auxiliary heads，但不能进入模型 prompt、在线 action 选择输入或策略 advantage group key 之外的可见特征。

## 10. 样本量与计算预算判断

### 10.1 哪部分合理，哪部分应削减

结论调整为：

- 每题固定 4 个 world roots（healthy + 3 abnormal slots）具有明确的对照与覆盖价值，数量合理；
- 固定 `2 replicas/world` 应删除，因为根部 branch 已经覆盖它，且 2 条本身不足以形成可靠的四样本优势；
- 每题 36～52 条最终 records 可能合理，但只能是由高价值 branch events 产生的结果，不能把它当成固定目标；
- 当前 `branch_probability=1.0`、`decision_regret_threshold=-1.0` 容易让低价值节点也展开，需要改为 score + token budget；
- 最重要的成本指标不是 records 数，而是“每个有效、非退化 advantage group 消耗的 generated tokens/GPU 秒”。

### 10.2 推荐的初始预算形态

不先把平均记录数写死为 5 或 6。先以统一分支结构做小规模消融：

1. 每题固定 4 个 primary world roots；
2. 每个 branch event `K=4` 起步；
3. 只对最高信息节点自适应升到 `K=8`；
4. 每题默认不超过 2 个额外 world realizations；
5. 分支一直扩展到题目级 generation-token budget 用尽，或下一 event 的预计边际信息收益低于阈值；
6. 初期禁止 branch-child 递归分支；
7. 36～52 条可作为难题/高信息题的允许区间或软上限，但不要求每题达到；
8. 简单题、动作已经一致或 reward 无方差的题应明显低于该数量。

建议对相同总 generation-token 预算比较：

```text
A. 4 roots + 只在根部做 K=4
B. 4 roots + top-node K=4，少量 K=8
C. 4 roots + 自适应额外 world variants + top-node K=4/8
D. 当前 4 worlds × 2 replicas + aggressive branch（诊断基线）
```

最终选择每 GPU 小时 reward/ANLS 提升最高、有效 advantage 最多且 Q calibration 改善的配置，而不是 records 最多的配置。

### 10.3 必须监控的边际价值

```text
valid_decision_group_rate
decision_group_size_histogram
distinct_canonical_action_count
nonzero_return_variance_group_rate
nonzero_advantage_group_rate
generated_tokens_per_valid_group
generated_tokens_per_nonzero_advantage
walltime_per_valid_group
reward_gain_per_1k_generated_tokens
ANLS_gain_per_GPU_hour
Q_ranking_gain_per_1k_generated_tokens
```

如果新增 branch 主要产生重复动作、相同 reward、协议错误或无效短轨迹，即使最终记录数看似很多，也应削减。

## 11. 计划中的代码修改步骤

本节仅规划，当前不执行代码改动。

### M1：统一身份与分组

1. 增加规范化的 `decision_group_id` 构造器；
2. 初始节点与中途节点使用同一 namespace；
3. 移除 replica 对 policy grouping 的语义，旧 `replica_id` 只保留为兼容/诊断字段；
4. rollout、FSDP、Megatron 只消费已验证的 canonical group ID；
5. 缺少关键身份字段时 hard fail，不允许 fallback 到 batch index。

### M2：四 World Root 与 Variant Sampler

1. 每题建立 H/L/F/C 四个默认 roots；
2. 每 slot 默认选择一个 latent world realization；
3. 按相关性、覆盖缺口、学习进展、探索与成本选择异常目标；
4. extra variants 各自建立新 root 和新 latent world ID；
5. 记录候选、得分、采样概率和固定 seed；
6. 禁止使用答案或当次最终 reward 选择 world。

### M3：统一 Checkpoint 与 Branch Sampler

1. 初始节点也创建可恢复 checkpoint；
2. primary path 中保存合格中间节点的完整 runtime digest；
3. 用允许的 online features 计算 branch priority；
4. 在题目级 token budget 下选择 top events；
5. 一次 batched sampling 生成 4 个或 8 个有效 on-policy continuations；
6. children 继承相同 latent world 和分支前 runtime；
7. 初期禁止递归 branch expansion。

### M4：Event-Level Advantage 与 Loss Mask

1. 只在 group size≥4 且 invariant 全部通过时计算 advantage；
2. 每个 branch event 独立求 baseline；
3. 中途分支只给 action/suffix token span 写 policy advantage；
4. 共享 prefix mask，不因 siblings 数量重复训练；
5. 多 event parent 使用各自 span，不复制全序列冲突 advantage；
6. optimizer packing 在 advantage 计算后进行，与分组彻底解耦。

### M5：隔离 Auxiliary Objectives

1. belief、Q、DVOI、risk、switch 和 pre-invariance 使用各自数据键；
2. 小于 4 条的 event 可用于 Q/risk 诊断，但不进入策略 advantage；
3. cross-world pairs 只进入明确建模 world 差异的 auxiliary loss；
4. Q bootstrap 与 terminal-return advantage 分开标记；
5. 分别报告 policy 与 auxiliary 的样本数、loss 和梯度贡献。

### M6：预算与日志

1. 记录 roots、world variants、candidate attempts、branch events、children 和 generated tokens；
2. 题目级预算在生成前检查，超限即停止扩展；
3. 不采用“先生成再丢弃”控制成本；
4. 记录 branch/world selection propensity；
5. 输出每种 world、每类 branch location 和 K=4/K=8 的独立收益。

## 12. 必须增加的测试

### 12.1 身份与分组测试

- 初始节点的 4 条 continuations 得到同一个 decision group ID；
- 同题同 world、不同中间 prefix 得到不同 group ID；
- 同一 `world_slot`、不同 `latent_world_id` 得到不同 group ID；
- 同 prefix、不同 runtime schedule state 得到不同 group ID；
- 同 event 的 primary/children 继承完全相同的分支前 runtime digest；
- 不同题目、文档、tool schema 或 reward version 不可同组；
- 缺失关键 ID 明确拒绝，不触发旧 fallback。

### 12.2 数值与 Loss 测试

- 4/8-way group 的 advantage 加权和接近 0；
- 向另一个 world 增加样本不改变原 group advantage；
- 改变 optimizer batch size 或样本顺序不改变逐 event advantage；
- FSDP 与 Megatron 输出一致；
- 小于 4 条的 event 不产生 policy loss；
- 中途 branch 的共享 prefix loss mask 全为 0；
- 多 branch-event parent 的各 token span 只消费所属 event advantage；
- root branch 的结果与等价旧 4-replica 数值构造一致。

### 12.3 World 与 Branch Sampler 测试

- 每题默认正好建立 H/L/F/C 四个 roots；
- extra abnormal target 产生新的 latent world/root，不加入旧组；
- world selection 不读取 answer、answer page 或当次最终 reward；
- branch selection 不读取模拟器 hidden world label；
- K=4/K=8 candidates 都来自独立采样，并报告 canonical action diversity、完整轨迹去重率和 utility 方差；
- fixed seed 可复现 world、node 和 action selection；
- token budget 在生成前生效；
- 记录数满足 `N_final = R + Σ(K_e-1)`；
- candidate attempts 和 generated tokens 单独统计，不被最终记录数掩盖。

### 12.4 真实 Rollout 验收

对固定题目集同时运行 4 类 world：

- `cross_content_group_count == 0`；
- `cross_question_group_count == 0`；
- `cross_latent_world_group_count == 0`；
- `cross_prefix_group_count == 0`；
- `undersized_policy_group_count == 0`；
- 所有 policy groups 至少有 4 条独立有效 continuations，且不存在人工复制记录；
- terminal 与 bootstrapped groups 分开；
- branch 事件、belief/Q/reward 事件持续产生；
- policy loss、非零梯度与权重同步正常；
- 四类 world 的 reward、ANLS、completion、tool success 和恢复能力分别报告。

## 13. 旧 Rollout 与当前训练的处理

已观察的 rollout 3 有 72 条有效记录、2 道原始题和 7 个宽泛 advantage groups，部分大组混入多个 `world_id`。因此旧 run 可以证明生成、loss、梯度和权重同步链路在工作，但不能证明策略优势已经满足严格同 world/same-node 要求。

处理原则：

1. 当前运行中的训练进程不在本规划阶段停止、覆盖或重启；
2. 保留现有日志、rollout 和 checkpoint，标记为 grouping diagnostic baseline；
3. 旧 rollout 不进入修复后的严格策略 replay；
4. 实现修复后从未受错误跨 world policy gradient 污染的基础/Stage-A checkpoint 启动；
5. 先做固定种子小规模消融，再决定完整训练的 K、extra-world 和 token budgets；
6. 最终 200 题评估使用固定 world 分布和最终 checkpoint，不让 eval 数据进入 optimizer。

## 14. 完成定义

只有同时满足以下条件，方案实施后才算完成：

1. replica 不再是独立 policy grouping 机制，根部与中途节点统一为 decision events；
2. 任一 policy advantage group 只含一个 question、latent world、prefix 和 runtime state；
3. 所有 policy groups 至少有 4 个有效 on-policy siblings；
4. 不同 branch locations、world variants 和题目无法静默混组；
5. 中途分支只训练对应 action/suffix，共享 prefix 不重复贡献 loss；
6. 四个 world slots 正常覆盖并分别发挥 healthy 对照、local fault、shared fault 和 change adaptation 的作用；
7. belief/Q/DVOI/risk 的输入不泄露 hidden world/答案，并有独立校准指标；
8. 36～52 条等较大展开只在高信息题上出现，且 generated-token 边际收益可解释；
9. 与当前 aggressive baseline 相比，在相同 generation-token/GPU-hour 下 reward、ANLS 或恢复能力有稳定提升；
10. FSDP/Megatron、checkpoint、权重同步和最终评估全链路通过。
