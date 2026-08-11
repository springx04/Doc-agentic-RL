# BayesTool-RL 最终实现方案

以下方案完整保留方法文件中的非平稳工具世界、分层隐变量、工具质量向量、在线过滤与离线平滑、HBD 后验蒸馏、配对工具世界、信念切换、信息不足阶段不变性、贝叶斯决策后悔、DVOI、共享前缀分支、sibling advantage、元回合、Persistent Consensus–Probe–Reopen、稳健提交、停止与拒答等全部设计，不把 BayesTool-RL 简化成过程奖励或普通 GRPO 奖励塑形。

当前项目已经有严格多轮工具协议、导航状态、证据状态、图像输入、动作 token span、基础设施错误隔离、GRPO/step-wise advantage 和动作级负信号，因此不需要重写 agent 或训练框架。实现重点应放在工具执行包装层、rollout 状态、配对世界采样、信念模块、分支数据构造和 Bayes-ARPO advantage 上。现有 `generate_with_retool.py` 已经维护完整多轮上下文、工具轨迹、视觉输入、动作 token span 和丰富 metadata；`reward_func` 已经隔离基础设施错误；slime 侧已经能够传输动作级奖励与 step span，并在 Megatron loss 中消费它们。

---

## 一、最终工程架构

最终系统划分为四层。

第一层是现有文档工具 agent。八个工具、严格 `<tool_call>`/`<final>` 协议、页面导航、证据抽取、多模态输入、最终答案 guard 全部保留。

第二层是 Tool-World Environment。它位于 `tool_registry.execute()` 与 agent 可见 observation 之间，对真实工具输出施加会话级、上下文级、共享故障级和阶段级扰动，并保留隐藏的真实世界状态和干净输出用于训练监督。策略永远只能看到扰动后的 observation。

第三层是 Tool-World Belief。一个小型时序模型维护会话信念、页面或区域信念、工具族共享信念、状态切换概率和每个工具五维质量分布，并向 Qwen3-VL 提供固定格式的 `<tool_belief>` 结构化接口。

第四层是 Bayes-ARPO。它继续使用 slime 的 clipped GRPO/PPO policy loss，但把普通同题组归一化改为共享前缀 sibling advantage，并增加独立的信念切换与 pre-invariance 辅助 minibatch。Belief 网络使用独立的小型优化器训练，Qwen3-VL 使用现有 Megatron actor 优化器训练，两者采用块坐标优化，不要求跨框架反向传播。

---

## 二、文件级改动

为控制改动面，只新增一个 `bayestool` 包，不拆出大量零散服务。

### 新增文件

| 文件 | 责任 |
|---|---|
| `toolcall-rl/bayestool/config.py` | 所有 BayesTool 参数、阈值和训练阶段配置 |
| `toolcall-rl/bayestool/schema.py` | `ToolWorldSpec`、`BeliefSnapshot`、`WorldEvent`、`BranchRecord` 等 dataclass |
| `toolcall-rl/bayestool/world.py` | 工具世界采样、工具结果缓存、扰动、状态切换和成本注入 |
| `toolcall-rl/bayestool/belief.py` | 特征抽取、filter、smoother、observation predictor、change detector |
| `toolcall-rl/bayestool/decision.py` | 粒子采样、Q 估计、共识、DR、DVOI、CVaR、reopen 和停止风险 |
| `toolcall-rl/bayestool/training.py` | utility、sibling advantage、支持有效配对、switch/pre-inv 辅助数据 |
| `toolcall-rl/bayestool/meta_episode.py` | 同文档多问题元回合和 persistent session belief |
| `toolcall-rl/train_bayestool_belief.py` | 阶段 A 的 filter/smoother/Q head 训练入口 |
| `toolcall-rl/build_bayestool_data.py` | coupling ID、元回合、固定评测世界构造 |
| `toolcall-rl/retool_qwen3_vl_4b_bayestool_rl.sh` | 4B、四卡训练入口 |
| `toolcall-rl/retool_qwen3_vl_8b_bayestool_rl.sh` | 8B、四卡训练入口 |

### 修改文件

`generate_with_retool.py` 是主要集成点，但保留其现有协议、导航、证据和多模态逻辑。需要完成以下改动：

1. 将当前单问题生成主体抽成 `_run_question(...)`。
2. 初始化 `WorldRuntime`、`BeliefRuntime` 和 `DecisionController`。
3. 在每个工具结果返回后执行扰动、特征抽取、信念更新和 surprise 检测。
4. 每轮把结构化任务状态和信念添加为环境 user token。
5. 支持从一个共享前缀返回多个 sibling `Sample`。
6. 支持元回合依次运行多个问题并传递 session belief。
7. 在 metadata 中导出 BayesTool 训练字段，但严格阻止隐藏真实世界进入 prompt。

其他改动如下：

- `tool_protocol.py`：增加严格 `<abstain>...</abstain>` 动作；仍然每轮只允许一个动作。
- `document_reward.py`：识别 abstention，区分正确回答、合理拒答和无必要拒答。
- `rl_data_preprocess.py`：写入 `coupling_id`、`document_hash`、`meta_episode_id` 和问题顺序。
- `slime/slime/ray/rollout.py`：增加 `bayes_grpo` 分组、sibling advantage 和辅助 bundle 传输。
- `slime/slime/backends/megatron_utils/loss.py`：把 `bayes_grpo` 纳入 GRPO policy loss，并增加 switch/pre-inv 辅助 loss。
- `slime/slime/utils/arguments.py`：增加 BayesTool 参数。
- 训练 actor 中唯一调用 `policy_loss_function` 的位置：增加独立的 Bayes auxiliary microbatch forward，并在同一次 optimizer step 前累积梯度。不要建立第二套 Qwen 训练框架。

当前 Qwen3-VL hook 已经复用 `generate_with_retool.generate/reward_func`，所以新实现继续沿用这个入口即可。

---

## 三、工具世界的数据结构

### 3.1 `ToolWorldSpec`

```python
@dataclass
class ToolWorldSpec:
    coupling_id: str
    world_id: str
    seed: int

    session_state: SessionStateSpec
    tool_states: dict[str, ToolQualitySpec]
    shared_factors: dict[str, SharedFactorSpec]
    context_rules: list[ContextRule]
    regime_schedule: list[RegimeSegment]
```

每个 `ToolQualitySpec` 包含：

```python
@dataclass
class ToolQualitySpec:
    availability: float
    semantic_accuracy: float
    structure_fidelity: float
    calibration_temperature: float
    calibration_bias: float
    relative_cost: float
    latency_scale: float
```

其中前四个质量维度限制在 `[0,1]` 或相应合理范围内，成本使用相对健康世界的倍率。

共享故障族固定为：

```text
render_core:
    render_page, crop_region, zoom_region, ocr_region, detect_layout

text_core:
    parse_document, ocr_region

structure_core:
    detect_layout, extract_table, chart_to_table
```

一个工具可以同时属于多个共享族。例如 `ocr_region` 可能因为渲染链路故障，也可能因为 OCR/text backend 故障而退化。

### 3.2 上下文规则

```python
@dataclass
class ContextRule:
    scope: Literal["document", "page", "region", "content_type"]
    tool_names: tuple[str, ...]
    page_numbers: tuple[int, ...] | None
    region: tuple[float, float, float, float] | None
    content_types: tuple[str, ...] | None
    overrides: ToolQualitySpecPatch
```

区域统一使用归一化坐标 `[0,1]`。内容类型只允许 `text/table/chart/figure/formula/mixed`。

### 3.3 非平稳阶段

```python
@dataclass
class RegimeSegment:
    start_call: int
    end_call: int | None
    transition: Literal["stable", "abrupt", "linear"]
    tool_overrides: dict[str, ToolQualitySpecPatch]
    shared_overrides: dict[str, SharedFactorSpecPatch]
```

运行时通过当前已执行工具调用数确定阶段。`abrupt` 在边界一步切换，`linear` 在区间内逐步插值。

---

## 四、耦合工具世界生成

每一个原始 `(D,q,y)` 不复制文档和答案，只添加 `coupling_id`。在 rollout 时根据 `sample.index` 确定工具世界：

```python
world_slot = sample.index % worlds_per_prompt
replica_id = (sample.index // worlds_per_prompt) % replicas_per_world
world_seed = hash(coupling_id, rollout_id, world_slot)
```

默认使用：

```yaml
worlds_per_prompt: 4
replicas_per_world: 2
n_samples_per_prompt: 8
```

四个 world slot 应覆盖：

1. 健康世界；
2. 单工具或单内容域退化；
3. 共享工具族故障；
4. 非平稳世界，包含 abrupt 或 gradual change。

训练世界类型的默认采样概率为：

```yaml
healthy: 0.20
single_tool_degradation: 0.25
context_degradation: 0.20
shared_family_fault: 0.20
abrupt_change: 0.10
gradual_change: 0.05
```

健康世界的基础范围：

```text
availability:       [0.99, 1.00]
semantic_accuracy: [0.95, 1.00]
structure_fidelity:[0.95, 1.00]
relative_cost:     [0.90, 1.10]
```

退化维度从 `[0.35,0.85]` 中采样，完全不可用状态只占退化世界的一小部分，避免把任务简化成二值工具故障识别。

### 4.1 工具扰动规则

所有扰动必须满足两个硬约束：

- 扰动器不得读取 `label`、`answers`、`answer_page`、`answer_bbox` 或目标答案字符串；
- 原始 PDF 和标准答案永远不变，只改变 agent 的 observation channel。

具体规则：

- `parse_document`：随机段落遗漏、局部字符替换、顺序扰动、页面漏检和虚拟延迟。
- `render_page`：模糊、降采样、局部遮挡、压缩噪声、旋转偏差。
- `crop_region/zoom_region`：bbox 平移、缩放误差、边缘截断和分辨率退化。
- `ocr_region`：字符删除、替换、数字混淆、行顺序错误、置信度错校准。
- `detect_layout`：bbox jitter、区域漏检、区域重复、类型误分类。
- `extract_table`：行列遗漏、行列交换、单元格合并错误、结构降级为文本。
- `chart_to_table`：系列遗漏、刻度偏差、类别错位、数值解析噪声。

对于数值扰动，必须使用目标无关规则，例如对全部数值按固定概率扰动，不能只改答案所在单元格。

### 4.2 干净结果缓存

真实工具只执行一次：

```text
cache_key =
    document_hash
    + canonical_tool_name
    + canonical_arguments
    + tool_backend_version
```

不同世界共享干净结果，再分别产生 world-specific observation。图像扰动结果写到：

```text
tool_outputs/bayestool/{coupling_id}/{world_id}/{call_id}/
```

禁止覆盖原始渲染图。

`WorldRuntime.transform_result()` 返回：

```python
observed_result: str
world_event: WorldEvent
hidden_supervision: ToolStateLabel
```

只有 `observed_result` 可以进入模型上下文。后两项只能写入 `metadata["bayes_supervision"]`。

真实基础设施错误与注入世界故障必须显式区分：

```text
failure_origin = "real_infrastructure"
failure_origin = "world_injected"
failure_origin = "model_action"
```

`world_injected` 是合法 POMDP 观测，必须参与 RL；只有真实基础设施错误继续沿用当前 `valid_for_rl=False` 的过滤逻辑。

---

## 五、任务状态和信念接口

### 5.1 任务状态

不额外训练大型 `TaskEncoder`，直接复用现有 `navigation_state`，构造确定性的 `TaskStateView`：

```python
@dataclass
class TaskStateView:
    question_type: str
    current_page: int | None
    visited_pages: tuple[int, ...]
    unvisited_page_count: int | None
    table_candidate_pages: tuple[int, ...]
    supporting_pages: tuple[int, ...]
    evidence_sufficient: bool
    visual_input_required: bool
    remaining_tool_budget: int
    last_tool: str | None
    last_result_status: str | None
```

这相当于可执行的 \(m_t\)。当前仓库已有这些导航和证据字段，因此主要是增加一个稳定的序列化层，而不是重新建立页面管理系统。

### 5.2 策略可见的信念

```python
@dataclass
class BeliefSnapshot:
    version: int
    step: int
    session_probs: tuple[float, ...]
    shared_family_probs: dict[str, tuple[float, ...]]
    regime_probs: tuple[float, ...]
    change_probability: float
    tool_quality: dict[str, ToolQualityPosterior]
    posterior_entropy: float
    ood_score: float
```

每个工具后验为：

```python
@dataclass
class ToolQualityPosterior:
    availability_mean: float
    availability_std: float
    semantic_mean: float
    semantic_std: float
    structure_mean: float
    structure_std: float
    calibration_mean: float
    calibration_std: float
    cost_mean: float
    cost_std: float
```

模型每轮看到：

```xml
<task_state>
{"question_type":"table", ...}
</task_state>
<tool_belief>
{
  "change_probability": 0.18,
  "tools": {
    "extract_table": {
      "availability": [0.91, 0.07],
      "semantic": [0.82, 0.11],
      "structure": [0.48, 0.19],
      "cost": [1.12, 0.15]
    }
  }
}
</tool_belief>
```

数值保留两位小数，按固定工具顺序序列化，整个 block 控制在约 1200 token 以内。该 block 属于环境 observation，`loss_mask=0`。

原始 observation 中保留 agent 实际获得的文本、表格、图像路径、status 和 error，但删除以下字段：

```text
world_id
true_quality
corruption_type
clean_result
clean_confidence
ground_truth_state
answer_page
answer_bbox
```

这使策略获得内容证据和实际失败现象，但不会直接看到模拟器的隐状态标签。

---

## 六、Tool-World Filter

### 6.1 输入特征

采用命名特征而不是散乱的魔法下标，最终由 `FEATURE_NAMES` 固定成 96 维向量。特征组如下：

- 8 维工具 one-hot；
- 6 维 status one-hot：`ok/partial/error/timeout/invalid/empty`；
- 8 维错误族 one-hot；
- 16 维通用统计：延迟、字符数、token 数、图像数、页面数、截断比例等；
- 16 维结构特征：bbox 合法性、行列数、schema 完整性、重复率等；
- 16 维语义与跨工具一致性：OCR/parse agreement、table/text agreement、数值一致率等；
- 16 维任务上下文：问题类型、当前页、剩余预算、是否重复调用、信息增益等；
- 10 维历史特征：连续失败次数、最近工具族、最近 surprise、调用阶段等。

所有连续值先 `log1p` 或归一化，再裁剪到 `[-5,5]`。缺失值使用零，同时增加相应 missing flag。

### 6.2 网络结构

Filter 使用一个很小的 PyTorch 模型，不修改 Qwen3-VL 主体：

```text
Observation encoder:
    Linear(96, 256)
    GELU
    LayerNorm(256)

Session filter:
    GRUCell(256, 256)

Context filter:
    GRUCell(256, 256)
    state key = document_hash/page/region_bucket

Shared-family filter:
    one GRUCell(256, 128) per shared family

Tool embedding:
    Embedding(8, 32)
```

输出分布：

- `session_state`：4 类，`healthy/degraded/overloaded/outage`；
- `regime_state`：3 类，`stable/abrupt_transition/gradual_transition`；
- 每个共享族：3 类，`healthy/degraded/down`；
- availability、semantic、structure、calibration：Beta 分布；
- cost：LogNormal 分布；
- change probability：Bernoulli。

Beta 参数通过 `softplus(raw)+1.0` 产生，避免退化到非法参数。

### 6.3 离线 smoother

Smoother 使用相同 observation encoder，加一个双向 GRU：

```text
BiGRU(input=256, hidden=256, num_layers=1)
```

输出头与 filter 相同。Smoother 只在训练阶段运行，完整轨迹可见；filter 只能看到前缀。

HBD loss 对所有分布求 KL：

```text
categorical posterior: categorical KL
Beta quality posterior: Beta KL
cost posterior: LogNormal KL
change probability: Bernoulli KL
```

smoother 输出全部 `detach`。

### 6.4 Observation predictor

预测下一次调用的以下离散观测：

```text
status:              6 classes
latency_bin:         8 classes
information_gain:    5 classes
semantic_agreement:  5 classes
schema_valid:        binary
image_valid:         binary
```

输入为：

```text
combined belief hidden
+ candidate tool embedding
+ task feature projection
```

预测惊异是各实际观测分量负对数似然之和，而不是对原始长文本做 token-level language modeling。

### 6.5 Filter 总损失

阶段 A 使用：

\[
L_{\text{belief}}
=
1.0L_{\mathrm{HBD}}
+
0.5L_{\mathrm{transition}}
+
1.0L_{\mathrm{obs}}
+
0.2L_{\mathrm{calibration}}.
\]

校准项对 availability、semantic correctness、structure correctness 和 change point 使用 Brier loss。

Filter 默认配置：

```yaml
feature_dim: 96
session_hidden: 256
context_hidden: 256
shared_hidden: 128
learning_rate: 3.0e-4
weight_decay: 1.0e-4
batch_size: 256
sequence_length: 16
gradient_clip: 1.0
early_stop_patience: 5
```

真实日志没有完整隐状态标签时，只计算可获得的 observation、status、cross-tool agreement 和 calibration 代理损失；不得伪造真实状态监督。

---

## 七、信念条件策略的具体接入

不修改 Qwen3-VL attention、视觉编码器或输出头。策略仍然是当前模型，只是每个 action 生成前都显式接收：

\[
\pi_\theta(a_t\mid \texttt{TaskStateView},\texttt{BeliefSnapshot}).
\]

`generate_with_retool.py` 中每次工具结果处理顺序固定为：

```text
1. 执行真实工具或读取 clean cache
2. WorldRuntime 生成 observed result
3. 仅用 observed result 更新 navigation_state
4. 提取 ObservationFeatures
5. BeliefRuntime.update(...)
6. 计算 predictive surprise
7. 按需 reopen
8. 构造 task_state + tool_belief
9. 编码成新的 user turn
10. 请求下一轮 action
```

禁止先用 `true_world` 更新 belief，也禁止用干净工具结果更新线上 filter。

为验证策略没有忽略信念，在训练日志中增加 `belief_intervention_compliance`：

```text
固定 TaskStateView 和候选动作集；
分别注入 b_u 和 b_v；
计算两个候选动作分布的 JS 和 argmax；
记录动作是否按目标世界切换。
```

---

## 八、粒子、动作价值、共识和后悔

### 8.1 后验粒子

默认从 Beta、LogNormal 和 categorical 后验中采样：

```yaml
posterior_particles: 8
```

粒子只包含工具世界状态，不复制 Qwen 参数。

### 8.2 候选动作

每个分支点最多保留四个 canonical action：

```yaml
max_action_candidates: 4
```

来源为同一共享前缀下的独立 policy sample。默认采样四次：

```text
temperature = 0.7
top_p = 0.95
different sampling seeds
```

每个输出经过当前严格 parser、参数校验和 canonicalization。重复 action 去重。

canonical action key：

```text
final:<normalized_answer>
abstain:<normalized_reason>
tool:<tool_name>:<sorted_normalized_arguments>
```

候选必须由 policy 实际生成，这样候选 action 已有 rollout token、old log-prob 和合法 importance ratio，不在 policy gradient 中训练环境强行插入的工具调用。

### 8.3 Q 估计器

增加一个小型 `BayesQHead`，它只为分支决策提供值估计，不作为 PPO critic：

```text
task feature projection: 32
world particle projection: 64
action feature projection: 64
budget/phase features: 8

MLP:
    Linear(168, 256)
    GELU
    Linear(256, 128)
    GELU
    Linear(128, 2)
```

输出 `return_mean` 和 `return_log_variance`。

Action feature 包含：

- action kind one-hot；
- tool one-hot；
- page number；
- bbox area；
- expected cost；
- 是否重复；
- 是否诊断性工具；
- 规范化 action 字符串的固定 hash embedding。

Q head 的监督来自实际 sibling continuation utility：

```text
target = complete branch trajectory utility
loss = Gaussian NLL
```

最初没有 Q head 时，阶段 C 前几个 rollout 直接使用 sibling Monte Carlo return；收集足够分支后再启用 Q head。

### 8.4 共识与 DR

对每个粒子计算候选动作 argmax，严格按方法文件计算 \(C_t\) 和 \(\mathrm{DR}_t\)。

默认阈值：

```yaml
consensus_threshold: 0.75
decision_regret_threshold: 0.08
```

训练时仅在：

```text
DR > 0.08
posterior OOD score < 0.15
remaining budget >= 2
```

时允许产生 sibling branches。

为控制计算量，再使用一个无偏随机 gate：

```yaml
branch_probability_when_eligible: 0.25
branch_horizon: 3
max_siblings: 4
```

所有 sibling 共享已有 Qwen prefix、navigation snapshot、belief snapshot 和 clean tool cache，只复制可变环境状态。

---

## 九、DVOI 的实现

诊断动作候选不由固定工具名单直接决定，而从候选动作中筛选：

```text
动作调用工具；
预期不直接回答任务；
其 observation predictor 的结果分布能够改变后续候选排序。
```

对每个候选 probe：

1. 从 observation predictor 构造最多六个高概率观测 hypothesis；
2. 对每个 hypothesis 调用 `BeliefRuntime.hypothetical_update()`；
3. 重新采样粒子并计算下一步 DR；
4. 计算期望 DR；
5. 扣除虚拟工具成本和剩余预算成本。

```yaml
max_observation_hypotheses: 6
dvoi_minimum: 0.0
```

只有 `DVOI > 0` 才执行 probe。

工具成本使用虚拟成本，不应为了模拟延迟而在训练时真的 `sleep`。真实 latency 只在评测真实工具世界时记录。

---

## 十、预测惊异和 Reopen

每次工具结果后计算：

\[
S_t=-\log p_\phi(o_t\mid h_{t-1},a_{t-1},b_{t-1}).
\]

默认阈值：

```yaml
local_surprise_threshold: 4.0
family_surprise_threshold: 6.0
global_surprise_threshold: 8.0
```

重新打开不直接把信念清零，而是与先验混合：

```text
local reopen:
    b_context ← 0.5 b_context + 0.5 prior_context

family reopen:
    b_family  ← 0.3 b_family  + 0.7 prior_family

global reopen:
    b_session ← 0.1 b_session + 0.9 prior_session
    clear all local context states
```

触发规则：

- 单页或单区域异常且其他工具正常：local；
- 同一 shared family 内两个相关工具连续异常：family；
- 不同工具族连续出现高 surprise，或 change head 给出 `P(change)>0.8`：global。

每次 reopen 写入：

```python
{
    "level": "local|family|global",
    "surprise": float,
    "cause_tools": [...],
    "belief_before": ...,
    "belief_after": ...
}
```

---

## 十一、轨迹效用

BayesTool 模式下不能继续使用当前“页面访问 +0.5、工具选择 +0.2”一类正向过程奖励，也不启用 PRM step score。当前这些逻辑保留给 baseline，但 `--bayestool-enable` 时走单独分支。

定义：

```text
S_task = 2 * quality - 1
```

其中 `quality` 直接复用当前 `compute_document_reward` 的 `[0,1]` 答案质量。

归一化成本：

\[
C=
0.40\frac{N_{\mathrm{call}}}{B}
+
0.25C_{\mathrm{latency}}
+
0.20C_{\mathrm{text\ token}}
+
0.15C_{\mathrm{image\ token}}.
\]

低效成本：

\[
I=
\min\left(
1,
\frac{
N_{\mathrm{duplicate}}
+N_{\mathrm{no\ gain}}
+N_{\mathrm{unnecessary}}
}{
\max(1,N_{\mathrm{call}})
}
\right).
\]

失败成本只统计 agent 可归责错误：

```text
protocol error
invalid arguments
premature final
unsupported final
把失败工具结果当成有效证据
预算耗尽但没有回答或拒答
```

注入世界故障本身不计入 \(F\)，真实基础设施错误直接排除出 RL。

默认：

\[
U=S_{\mathrm{task}}-0.15C-0.20I-0.35F.
\]

utility 不在 reward function 内先裁剪到 `[-1,1]`，避免正确但成本不同的轨迹被压成相同分数。只在日志可视化时额外记录 clipped utility。

当前 `reward_func` 的基础设施错误隔离与 reward consistency audit 应继续保留。

---

## 十二、Bayes-ARPO 与 sibling advantage

新增 estimator 名称：

```text
--advantage-estimator bayes_grpo
```

也可以在论文和日志中称其为 Bayes-ARPO，但代码名称统一为 `bayes_grpo`，避免与其他 ARPO 实现混淆。

归一化 key 按优先级选择：

```python
if sibling_group_id is not None:
    baseline_key = sibling_group_id
else:
    baseline_key = (coupling_id, world_id)
```

同一个 world 的两个 stochastic replica 可以形成基本组；发生分支时，同一共享前缀的 sibling 使用更精确的组。

\[
A_j=R_j-\sum_l\bar w_lR_l.
\]

默认只减均值，不做标准差除法：

```yaml
grpo_std_normalization: false
```

原因是两条或三条 sibling 的标准差估计不稳定。

`loss.py` 中把 `bayes_grpo` 与 `grpo` 一样广播 advantage 到对应 policy action token，并复用现有：

- clipped importance ratio；
- KL loss；
- entropy；
- action-level rejected token override；
- context parallel slicing。

现有 GRPO 和 step-wise 实现可以直接复用大部分逻辑。

---

## 十三、信念切换与 pre-invariance 辅助 loss

这两个目标不能伪装成 scalar reward，必须实现真正的 action log-prob loss。

### 13.1 支持有效配对

每个可配对状态保存：

```python
content_signature
belief_snapshot
candidate_actions
candidate_returns
posterior_ood_score
first_distinguishing_event_step
```

`content_signature` 由以下内容 hash：

```text
question
question_type
已访问页面集合
规范化语义证据摘要
剩余预算
当前任务阶段
```

工具状态、错误文本、belief 和 world ID 不进入 signature。

训练阶段的配对条件：

```text
same coupling_id
same content_signature
both OOD scores < 0.15
0.10 <= JS(b_u, b_v) <= 0.80
best action margin in each world >= 0.05
a_u* != a_v*
```

语义等价性可以在离线 pair builder 中利用隐藏 clean result 检查，但 clean result 不进入策略输入。

### 13.2 Action sequence score

候选 action 是多 token 字符串，因此采用长度归一化 sequence log-prob：

\[
\ell_\theta(a\mid x)
=
\frac{1}{|a|}
\sum_{i=1}^{|a|}
\log\pi_\theta(a_i\mid x,a_{<i}).
\]

### 13.3 Switch bundle

每个 switch pair 构造四条 teacher-forced sequence：

```text
prompt(m, b_u) + a_u*
prompt(m, b_u) + a_v*
prompt(m, b_v) + a_v*
prompt(m, b_v) + a_u*
```

四条 sequence 构成不可拆分的 `switch_bundle_id`。严格实现方法文件中的双向排序公式：

\[
L_{\mathrm{switch}}
=
-\log\sigma[
\ell_u(a_u)-\ell_u(a_v)
+
\ell_v(a_v)-\ell_v(a_u)
].
\]

### 13.4 Pre-invariance bundle

只在第一个可区分 observation 之前构造。要求：

```text
true worlds differ
observed prefixes still identical
belief JS < 0.02
candidate action set identical
```

对同一个最多四动作候选集分别计算：

\[
p_u(a)=\operatorname{softmax}(\ell(a|m,b_u)/0.5),
\]

\[
p_v(a)=\operatorname{softmax}(\ell(a|m,b_v)/0.5),
\]

然后使用候选动作分布上的 JS divergence。

### 13.5 训练管线

`sample.metadata["bayes_aux_records"]` 保存已经 tokenized 的 bundle。`rollout.py` 聚合这些记录，并保证：

- 一个 bundle 只分配给一个 DP rank；
- switch 四序列不得跨 microbatch；
- pre-inv 的全部候选序列不得跨 microbatch；
- `aux_micro_batch_size` 是四的倍数；
- 辅助序列不参与普通 GRPO reward normalization。

Actor 每两次 policy step 运行一次 auxiliary forward：

```yaml
aux_interval: 2
max_switch_bundles_per_rank: 8
max_preinv_bundles_per_rank: 8
switch_loss_weight: 0.20
preinv_loss_weight: 0.05
```

主 loss：

\[
L_\theta=
L_{\mathrm{BA-policy}}
+0.20L_{\mathrm{switch}}
+0.05L_{\mathrm{pre-inv}}
+0.01L_{\mathrm{KL}}.
\]

两个 forward 的梯度在同一次 optimizer step 前累积，不能使用独立 Qwen optimizer，也不能把 auxiliary loss 变成离线标签分类。

---

## 十四、贝叶斯预言机后悔

实现中不能搜索无限轨迹，因此使用方法文件已经定义的有限候选分支作为可计算预言机：

```text
oracle_utility(world z, state h)
    = max utility among candidate sibling continuations
```

记录：

```text
oracle_action
oracle_utility
policy_utility
relative_regret
```

Q head 和 branch target 使用相同 utility 定义。

相对后悔主要用于：

- Q head 训练；
- branch 触发分析；
- switch pair 筛选；
- 评测和论文指标。

默认不再额外把 regret 重复加进 reward，因为 utility 差和 sibling advantage 已经包含这部分信号。

---

## 十五、元回合

### 15.1 数据格式

同一 PDF 至少有两个问题时构造：

```json
{
  "meta_episode_id": "...",
  "document_path": "...",
  "questions": [
    {"prompt": "...", "label": "...", "metadata": {...}},
    {"prompt": "...", "label": "...", "metadata": {...}}
  ]
}
```

默认每个元回合包含 2–4 个问题。问题顺序在每个 epoch 内随机，但所有 coupled world replicas 使用相同顺序。

### 15.2 运行方式

`generate()` 遇到 `metadata["meta_questions"]` 后：

1. 采样一次 session world；
2. 运行问题 1；
3. 保存 `session_hidden` 和 shared-family hidden；
4. 清除 navigation、证据和 context hidden；
5. 运行问题 2；
6. 依次继续；
7. 返回一个 `list[Sample]`，每个问题仍是独立训练样本，但具有相同 `meta_episode_id/world_id`。

这利用了 RolloutManager 已能展平嵌套样本的现有行为，不需要在一条 Qwen response 中支持多个 `<final>`。

### 15.3 跨问题信用

只使用每题即时 utility 会惩罚问题 1 中对后续问题有价值的诊断调用，因此元回合使用 suffix return：

\[
G_i=
\sum_{j=i}^{N}
0.95^{j-i}U_j.
\]

问题 \(i\) 的 Bayes-ARPO reward 使用 \(G_i\)，而评测仍分别报告每题 \(U_i\)。

检测到全局状态切换后，session belief 按 global reopen 规则重置；局部页面异常不能清除整个 session belief。

---

## 十六、停止和拒答

训练一个小型 `AnswerRiskCalibrator`，使用验证集上的以下特征进行 logistic calibration：

```text
final evidence support
支持该答案的独立工具数量
来源工具 semantic posterior
来源工具 structure posterior
未访问页面比例
最近 surprise
答案自一致性
剩余预算
```

定义：

```text
R_stop     = predicted answer error probability
R_abstain  = 0.35
R_continue = min_a [normalized action cost + predicted future risk]
```

选择三者中风险最低者：

```text
stop    -> <final>...</final>
abstain -> <abstain>insufficient reliable evidence</abstain>
continue -> tool action
```

有标准答案且证据可正常获得时，无必要 abstain 应获得负 utility；工具世界使所有路径都不足以可靠回答时，合理 abstain 的 utility 高于编造答案，但低于正确回答。

---

## 十七、分阶段训练

### 阶段 A：工具世界识别预训练

输入：

- 当前 baseline/SFT agent 的真实工具日志；
- 对每条干净工具结果生成的多个 counterfactual world observations；
- 模拟状态标签；
- real-log proxy labels。

训练：

```text
filter
smoother
transition head
observation predictor
change detector
Q head warm-up
answer risk calibrator
```

数据按 document hash 划分 train/validation/test，不能把同一 PDF 的不同问题分到不同集合。

建议至少收集：

```text
50,000 条完整轨迹
或 200,000 个工具调用事件
```

不足时以工具调用事件数量为早期目标，并使用 validation early stopping。

### 阶段 B：单任务信念条件策略训练

- 每个样本只运行一个问题；
- world sampler 开启；
- filter 固定一个 checkpoint version；
- 使用结构化 belief block；
- 使用 trajectory utility；
- 使用 `bayes_grpo`；
- 暂不启用 meta episode；
- branch 比例限制为 0.10；
- switch/pre-inv 只使用高置信 pair。

这一步先让 Qwen 学会读取 belief，而不是立即承受全部分支复杂度。

### 阶段 C：配对世界信念切换训练

全部启用：

```text
M=4 coupled worlds
R=2 replicas
support-valid pairing
switch loss
pre-invariance loss
K=8 particles
decision regret branching
DVOI
sibling advantage
predictive surprise/reopen
Q head replay
```

这是论文核心训练阶段，应占总 Qwen RL update 的约 60%。

### 阶段 D：跨任务元策略训练

- 每个元回合 2–4 个问题；
- session belief 持久化；
- local task state 重置；
- 使用 suffix meta return；
- 允许跨问题状态切换；
- 健康世界和非平稳世界各占一半。

### Filter 的持续更新

Filter 与 Qwen 参数不共同反向传播。使用块坐标优化：

```text
固定 filter version，完成一个 rollout batch；
Qwen 执行若干 Bayes-ARPO update；
把新轨迹加入 belief replay；
CPU BeliefLearner 更新小模型；
只在下一个 rollout batch 开始时发布新 filter version。
```

每条 rollout metadata 必须写入 `belief_model_version`，同一 sibling group 不允许混用不同 filter version。

---

## 十八、四卡 A100 配置

当前 `retool_qwen3_4b_rl.sh` 默认是 8 卡、4 卡 actor 加 4 卡 rollout，并使用 TP=4，所以不能直接用于四卡环境。

最终拓扑固定为：

```text
GPU 0–1: Megatron actor，tensor parallel = 2
GPU 2–3: SGLang rollout engine，tensor parallel = 2
```

公共参数：

```bash
NUM_GPUS=4
ACTOR_GPUS=2
ROLLOUT_GPUS=2

--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--context-parallel-size 1
--rollout-num-gpus-per-engine 2

--advantage-estimator bayes_grpo
--use-kl-loss
--kl-loss-coef 0.01
--eps-clip 0.2
--eps-clip-high 0.28
--grpo-std-normalization false

--optimizer-cpu-offload
--use-dynamic-batch-size
--attention-dropout 0.0
--hidden-dropout 0.0
```

Qwen3-VL 官方模型名称应分别使用 `Qwen/Qwen3-VL-4B-Instruct` 和 `Qwen/Qwen3-VL-8B-Instruct`；两者均有官方多模态 processor 和 SGLang 使用方式。

不要继续 source 当前纯文本 `qwen3-4B.sh` 后仅替换 checkpoint。应新建 Qwen3-VL Megatron 配置，保留：

- `model_type=qwen3_vl`；
- Qwen3-VL vision tower；
- MRoPE；
- image token、vision start/end token；
- multimodal processor；
- `rope_theta=5000000`。

### A100 80GB 默认值

4B：

```yaml
rollout_batch_size: 8
n_samples_per_prompt: 8
max_tokens_per_gpu: 16384
rollout_max_context_len: 16384
rollout_max_response_len: 8192
branch_probability: 0.25
```

8B：

```yaml
rollout_batch_size: 4
n_samples_per_prompt: 8
max_tokens_per_gpu: 12288
rollout_max_context_len: 16384
rollout_max_response_len: 6144
branch_probability: 0.20
```

### A100 40GB 安全值

4B：

```yaml
rollout_batch_size: 4
max_tokens_per_gpu: 12288
```

8B：

```yaml
rollout_batch_size: 2
max_tokens_per_gpu: 8192
rollout_max_context_len: 12288
branch_probability: 0.15
```

四卡训练期间默认关闭本地 PRM GPU：

```bash
--prm-enable false
```

保留当前 PRM 实现用于 baseline 对照，但 BayesTool 主实验不得把 PRM step score 加进 utility，否则会重新退化为正向步骤奖励。

---

## 十九、默认配置

```yaml
bayestool:
  enabled: true

  worlds_per_prompt: 4
  replicas_per_world: 2

  posterior_particles: 8
  max_action_candidates: 4
  max_siblings: 4
  branch_horizon: 3
  branch_probability_when_eligible: 0.25

  consensus_threshold: 0.75
  decision_regret_threshold: 0.08
  max_observation_hypotheses: 6
  cvar_alpha: 0.20

  local_surprise_threshold: 4.0
  family_surprise_threshold: 6.0
  global_surprise_threshold: 8.0
  change_probability_threshold: 0.80

  utility:
    cost_weight: 0.15
    inefficiency_weight: 0.20
    failure_weight: 0.35

  losses:
    switch_weight: 0.20
    preinv_weight: 0.05
    kl_weight: 0.01

  pairing:
    max_ood_score: 0.15
    min_belief_js: 0.10
    max_belief_js: 0.80
    min_action_margin: 0.05
    preinv_max_js: 0.02

  auxiliary:
    interval: 2
    max_switch_bundles_per_rank: 8
    max_preinv_bundles_per_rank: 8

  meta:
    questions_per_episode_min: 2
    questions_per_episode_max: 4
    discount: 0.95
```

---

## 二十、必须增加的审计与测试

### 单元测试

新增：

```text
test_world_determinism.py
test_world_no_answer_leakage.py
test_tool_corruptors.py
test_belief_filter_shapes.py
test_hbd_loss.py
test_observation_surprise.py
test_reopen_levels.py
test_sibling_advantage.py
test_switch_loss.py
test_preinv_loss.py
test_support_valid_pairing.py
test_dvoi.py
test_meta_episode_persistence.py
test_bayestool_reward.py
```

关键断言：

1. 同一 seed 产生完全相同的 world 和 observation。
2. 修改答案 metadata 不得改变扰动结果。
3. 干净结果不得出现在 policy prompt。
4. `world_injected` failure 保持 `valid_for_rl=True`。
5. `real_infrastructure` failure 保持 `valid_for_rl=False`。
6. 同一个 sibling group 使用相同 prefix、world、belief version。
7. switch bundle 四条 sequence 不跨 rank 和 microbatch。
8. pre-inv 只在区分 observation 出现前生成。
9. 所有环境 token 的 loss mask 为零。
10. 所有 policy action token 的 rollout log-prob 与 token 数严格对齐。

### 集成测试

固定同一问题构造两个世界：

```text
世界 U：extract_table 可靠，render/OCR 成本高
世界 V：extract_table 结构损坏，render/OCR 可靠
```

必须验证：

- 未调用任何区分性工具前，两个世界的候选动作分布接近；
- 观察到结构损坏后，belief 明显分离；
- U 中偏好 `extract_table`；
- V 中偏好 `render/crop/ocr`；
- belief intervention 能在固定 task state 下切换动作；
- 交换 belief 而不交换任务状态时，动作按 belief 改变；
- 工具状态中途切换后，在有限调用内触发 reopen 并改变路径。

### 四卡 smoke test

```text
2 actor GPU + 2 rollout GPU
4 个 prompt
每 prompt 4 worlds
每 world 至少一个完整 rollout
至少一个 sibling group
至少一个 switch bundle
至少一个 image observation
完成一次 optimizer step
```

必须检查：

```text
没有第五个 CUDA process 持有模型权重
没有 sibling prefix 重复保存完整视觉 tensor
没有辅助 bundle 跨 DP rank
没有 NaN advantage
没有空 reward group
```

---

## 二十一、论文和实验应报告的指标

任务指标：

```text
ANLS / EM / accuracy
healthy-world performance
perturbed-world performance
non-stationary-world performance
```

工具世界指标：

```text
filter NLL
Brier score
ECE
change-point detection delay
session/context/shared state accuracy
```

决策指标：

```text
relative oracle regret
belief switch accuracy
pre-information JS
consensus calibration
positive-DVOI precision
unnecessary probe rate
reopen level accuracy
```

效率指标：

```text
tool calls
latency cost
image token cost
duplicate calls
no-information-gain calls
```

元策略指标：

```text
第一个问题与后续问题的平均工具调用差
session belief transfer gain
状态突变后的恢复调用数
```

论文消融开关必须全部实现，但默认主方法全部打开：

```text
without HBD
without switch loss
without pre-invariance
without DVOI
without regret branching
without reopen
without meta episode
without persistent session belief
```

这些开关只用于实验消融，不应形成另一个简化版主实现。

---

## 最终落地原则

实现中最关键的三条边界是：

第一，BayesTool-RL 的主要策略优化信号必须是完整轨迹净效用和 sibling-relative advantage，不能继续把“正确工具调用 +0.1”作为核心训练信号。

第二，工具世界真实状态、干净工具输出和扰动标签只能用于 filter、smoother、pair builder 和评测监督，不能进入 Qwen prompt、navigation state 或在线动作选择。

第三，Qwen3-VL、ToolWorldFilter 和 BayesQHead 可以使用不同优化器，但必须通过固定 belief snapshot、world version、branch record 和 block-coordinate schedule 组成一个完整训练系统；不得为了减少实现量而删除 HBD、switch、pre-invariance、DVOI、reopen 或 meta episode。
